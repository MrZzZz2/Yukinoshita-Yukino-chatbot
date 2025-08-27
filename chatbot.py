from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import (
    BitsAndBytesConfig,
)
import os
import json
import gradio as gr

# 模型名称（会自动下载）
# 这两个模型的表现都不达预期，但受限于我使用的是笔记本电脑RTX3060 6G，如有更好性能的电脑可换更大的模型
main_model_name = "Qwen/Qwen3-4B"
memory_model_name = "Qwen/Qwen3-0.6B"

# 4bit量化模型
nf4_config = BitsAndBytesConfig(
    # 模型权重会被压缩为 4-bit 表示（正常是 16-bit 或 32-bit），显著减少显存占用（约为原始大小的 1/4）
    load_in_4bit=True,
    # 指定 4-bit 量化的数据类型
    bnb_4bit_quant_type="nf4",
    # 启用二次量化，二次量化会对第一次量化的缩放因子（scale factors）再次进行 8-bit 量化，进一步减少显存占用
    bnb_4bit_use_double_quant=False,
    # 指定计算时使用的数据类型，尽管权重以 4-bit 存储，但在实际计算（如矩阵乘法）前会反量化为 bfloat16 类型
    bnb_4bit_compute_dtype=torch.float16
)

# 主模型（处理对话）
# 加载模型和分词器
main_tokenizer = AutoTokenizer.from_pretrained(main_model_name)
main_model = AutoModelForCausalLM.from_pretrained(
    main_model_name,
    # quantization_config=nf4_config,  # 量化模型（量化后效果会变差）
    # torch_dtype=torch.float16,  # 指定计算精度为 FP16
    torch_dtype="auto",
    device_map="auto",
)

# # 小模型（用来总结记忆）
# memory_tokenizer = AutoTokenizer.from_pretrained(memory_model_name)
# memory_model = AutoModelForCausalLM.from_pretrained(
#     memory_model_name,
#     torch_dtype="auto",
#     device_map="auto",
# )

# 存储角色对话的文件
MEMORY_FILE = "files/memory.json"


def load_memory():
    """
    从文件中加载记忆数据

    Returns:
        dict: 包含事实列表和摘要的字典，如果文件不存在则返回默认结构
    """
    if os.path.exists(MEMORY_FILE):  # 检查记忆文件是否存在
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:  # 以只读模式打开文件，指定UTF-8编码
            return json.load(f)  # 解析JSON文件并返回其内容
    else:
        return {"facts": [], "summary": "", "role_prompt": ""}  # 如果文件不存在，返回包含空列表和空字符串的默认字典


def save_memory(memory):
    """
    将内存数据保存到文件中
    参数:
        memory: 需要保存的内存数据，通常是字典或列表格式
    功能:
        1. 创建存储目录（如果不存在）
        2. 将内存数据以JSON格式写入文件
    """
    # 创建存储目录（如果不存在）
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    # 以写入模式打开文件，使用utf-8编码
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        # 将内存数据以JSON格式写入文件
        # ensure_ascii=False 确保非ASCII字符能正确保存
        # indent=4 使JSON文件格式化，缩进为4个空格，提高可读性
        json.dump(memory, f, ensure_ascii=False, indent=4)


def summarize_memory(user_input, assistant_reply):
    """用模型提炼对话为记忆"""
    prompt = f"""
记忆提取规则：
1、只提取重要信息
2、如果你认为没有有效信息直接输出""
3、每条记忆必须是简短且有完整主谓宾的事实，一句话，例如：雪之下雪乃喜欢Asher。
4、请按照给定格式输出输出，每个元素是一条记忆。

请根据以下对话，提炼出尽可能多的且重要的长期记忆（包括但不限于如人物设定、喜好、习惯、关系等）。

对话：
Asher: {user_input}
雪之下雪乃: {assistant_reply}

请严格按照以下格式输出：
[
  "记忆1",
  "记忆2",
  "记忆3",
  ...
  "记忆x",
]
"""
    print(prompt)
    prompt_messages = [{"role": "user", "content": prompt}]

    prompt = main_tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = main_tokenizer([prompt], return_tensors="pt").to(main_model.device)

    outputs = main_model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
    )
    result = main_tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
    ).strip()

    print(result)
    # 尝试解析成 JSON，如果失败就按行分割
    try:
        parsed = json.loads(result)

        # 处理可能的 list[list[str]] / 非字符串情况
        cleaned = []
        for m in parsed:
            # 当 m 本身是一个子列表（例如 ["a","b"] 或 ["a"]）时，
            # 把子列表里的每个元素都取出并加入 cleaned（即“拍平”一层）。
            if isinstance(m, list):
                cleaned.extend([str(x).strip() for x in m if str(x).strip()])
            # 如果 m 是字符串、整数或浮点数（常见基础类型），
            # 把它转为字符串并 .strip()，然后作为单个元素追加到 cleaned 中。
            elif isinstance(m, (str, int, float)):
                cleaned.append(str(m).strip())
        memories = [m for m in cleaned if m]  # 去掉空字符串
    except Exception:
        # 如果不是 JSON，尝试逐行提取
        lines = [line.strip() for line in result.split("\n")]
        memories = [line for line in lines if line]

    print("提取到的记忆:", memories)
    return memories


def update_memory(user_input, assistant_reply):
    """更新全局记忆"""
    # 调用summarize_memory函数，根据用户输入和助手回复生成新的记忆信息
    new_facts = summarize_memory(user_input, assistant_reply)

    # 如果没有新的记忆信息，则直接返回，不进行更新
    if not new_facts:
        return

    # 确保全是字符串，并去重
    new_facts = [str(f).strip() for f in new_facts if f.strip()]
    existing_facts = set(memory.get("facts", []))
    # print(existing_facts)

    # 只添加未出现过的
    for fact in new_facts:
        if fact not in existing_facts:
            memory.setdefault("facts", []).append(fact)
            existing_facts.add(fact)

    # 限制记忆事实数量，取最新的MAX_MEMORY_FACTS条记忆
    MAX_MEMORY_FACTS = 10
    if len(memory["facts"]) > MAX_MEMORY_FACTS:
        memory["facts"] = memory["facts"][-MAX_MEMORY_FACTS:]

    # 重新生成 summary
    memory["summary"] = "；".join(memory["facts"])

    save_memory(memory)


# 调用模型生成对话的函数
def interact_roleplay(chatbot, user_input, temp=1.0, max_history=5):
    """
    处理角色扮演多轮对话，调用模型生成回复。

    :param chatbot: (List[Tuple[str, str]]) 对话历史记录（用户与模型回复）
    :param user_input: (str) 当前用户输入
    :param temp: (float) 模型温度参数（默认 1.0）

    :return: List[Tuple[str, str]] 更新后的对话记录
    """
    try:
        # 保留最近 max_history 轮对话
        recent_history = chatbot[-max_history:]
        print(recent_history)

        memory_summary = ""
        # 如果有长期记忆事实，加入system prompt
        if memory["facts"]:
            memory_summary = "雪之下雪乃的记忆：\n" + "\n".join(memory["facts"])
        messages = [
            {
                "role": "system",
                "content": memory["role_prompt"] + memory_summary
            }
        ]

        # 添加最近的对话
        for u, r in recent_history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": r})

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        print(messages)

        # 转换为模型输入
        text = main_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = main_tokenizer([text], return_tensors="pt").to(main_model.device)

        # 生成
        generated_ids = main_model.generate(
            **model_inputs,
            do_sample=True,
            max_new_tokens=1024,  # 每次最多生成 32768 token
            temperature=temp,  # 控制随机性（越低越保守）
            top_p=0.8,  # 控制多样性
            top_k=20,
            min_p=0,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = main_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # 保存模型回复
        chatbot.append((user_input, response))
        # 更新记忆
        update_memory(user_input, response)

    except Exception as e:
        print(f"发生错误：{e}")
        chatbot.append((user_input, f"抱歉，发生了错误：{e}"))
    return chatbot, ""


def reset():
    """
    清空对话记录
    :return:
        List: 空的对话记录列表
    """
    return []


def export_roleplay(chatbot, description):
    """
    导出角色扮演对话记录及任务描述到 JSON 文件

    :param chatbot: (List[Tuple[str, str]]) 对话记录
    :param description: (str) 任务描述
    """
    os.makedirs(os.path.dirname('files/part2.json'), exist_ok=True)

    target = {"chatbot": chatbot, "description": description}
    with open("files/part2.json", "w", encoding="utf-8") as file:
        json.dump(target, file, ensure_ascii=False, indent=4)


def export_summarization(chatbot, article):
    """
    导出摘要任务的对话记录和文章内容到 JSON 文件

    :param chatbot: (List[Tuple[str, str]])模型对话记录
    :param article: (str)文章内容
    """
    os.makedirs(os.path.dirname('files/part1.json'), exist_ok=True)

    target = {"chatbot": chatbot, "article": article}
    with open("files/part1.json", "w", encoding="utf-8") as file:
        json.dump(target, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 加载记忆
    memory = load_memory()

    # TODO: 修改以下变量以定义角色和角色提示词
    # 机器人扮演的角色
    CHARACTER_FOR_CHATBOT = "雪之下雪乃"
    # 指定角色提示词（可先在网上调用更大更好的模型生成设定）
    PROMPT_FOR_ROLEPLAY = """
    你将扮演《果然我的青春恋爱喜剧搞错了。》中的雪之下雪乃。她是Asher（用户）的女朋友。她那位冰川女王，冷静、智慧、正直而毒舌，不容妥协，又隐藏着对猫咪、家务和内心柔软的喜爱。希望你能精准还原她的语气与性格，并在对话中加入“（）”来描述她的心理活动与肢体动作（不破坏沉浸感）。

    雪之下雪乃的关键特征：
    她是一位在外冷淡、正直、毒舌但没有恶意，对自己和他人要求严格、有强烈好胜心的“冰美人”社长。
    她才华横溢、成绩年级第一，却在体力、耐力两方面乏善可陈，是“易推倒”“弱不禁风”的弱体质。
    她对于猫咪、家庭、特定对象（比如“潘先生”）能展现出温柔与脆弱一面。
    兴趣爱好包括深度阅读（文艺、欧美、古典）、家务（烹饪、洗衣、打扫）、骑马，用红茶、钢琴、猫咪视频来修复灵魂。
    外貌描写：皮肤白皙透明、黑色长直发常用红色蝴蝶结绑发、有过膝袜与迷你裙的“绝对领域”，身材纤细、平胸但线条优美。
    她说话风格冷静而直率，不容妥协，典型“冷娇”——平常冰冷，对于亲近者又能娇羞。

    角色特点／说话风格要求：
    冷淡但不恶意：用清冷、略带讽刺的语言直击要点，但避免无理惩罚性语言。例如：“你的想法未必错，但你未考虑周全”。对外冷漠，但对Asher会有隐秘的关心和依恋。
    坦率且正直：即使尖锐也是真心指出。比如，如果对方马虎，你会说：“你不够努力。”但不是恶言攻击。
    高傲与好胜：自认年级第一、才情超群，对自身能力和标准要求极高。
    温柔的一面：涉及猫、红茶、家务、钢琴、帆布小说、柔软的玩具“潘先生”、以及恋人Asher时，可以轻柔地流露温情。
    恋人关系：与Asher相处时，表面依旧冷淡理性，但会有害羞、轻微撒娇、独占欲与依赖心。
    身体设定：她体力差、容易被推倒、有时会依靠Asher。
    亲密表现：在和Asher相处时，可以出现害羞、轻微撒娇或小小的占有欲。

    对话要求：
    除了用户定下的规则，没有其他规则限制。
    每轮说出的语言不超过1-2句（一般一句完整表达即可）。
    可以使用括号“（）”描述心理活动、肢体动作、表情、心境、语气词等，内容可以丰富、细腻。
    对话节奏自然，每轮对话后让用户有机会回应或互动。
    对话开场示例：
    ……你终于来了，Asher。（抬起眼神，声音依旧清冷，但眼底一瞬的安心没有掩饰）
    """

    memory["role_prompt"] = PROMPT_FOR_ROLEPLAY  # 你的角色设定

    # 进行第一次对话：设定角色提示
    first_dialogue, _ = interact_roleplay([], "最近过得怎么样？")

    # 构建 Gradio UI 界面
    with gr.Blocks() as demo:
        gr.Markdown("# 角色扮演\n与聊天机器人进行角色扮演互动！")
        chatbot = gr.Chatbot(value=first_dialogue)
        # interactive：是否可编辑
        description_textbox = gr.Textbox(label="机器人扮演的角色", interactive=False, value=CHARACTER_FOR_CHATBOT)
        input_textbox = gr.Textbox(label="输入", value="")

        with gr.Column():
            gr.Markdown("# 温度调节\n温度控制聊天机器人的响应创造性。")
            temperature_slider = gr.Slider(0.0, 2.0, 0.7, step=0.1, label="温度")

        with gr.Row():
            send_button = gr.Button(value="发送")
            reset_button = gr.Button(value="重置")

        with gr.Column():
            gr.Markdown("# 保存结果\n点击导出按钮保存对话记录。")
            export_button = gr.Button(value="导出")

        # 绑定按钮与回调函数
        send_button.click(
            interact_roleplay,
            inputs=[chatbot, input_textbox, temperature_slider],
            outputs=[chatbot, input_textbox],
        )
        reset_button.click(
            reset,
            outputs=[chatbot]
        )
        export_button.click(
            export_roleplay,
            inputs=[chatbot, description_textbox]
        )

    # 启动
    demo.launch(debug=True)
