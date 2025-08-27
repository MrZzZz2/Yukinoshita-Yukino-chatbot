import os
import json
from typing import List, Dict, Tuple
import openai
import gradio as gr

# # TODO: 设置你的 OPENAI API 密钥，这里假设 DashScope API 被配置在了 OPENAI_API_KEY 环境变量中
# OPENAI_API_KEY = "sk-b9a82446166643d78ee67a1df43ab1c0"
# # 不填写则默认使用环境变量
# if not OPENAI_API_KEY:
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
# client = openai.OpenAI(
#     api_key=OPENAI_API_KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
#
# try:
#     response = client.chat.completions.create(
#         model="qwen-plus-0806",
#         messages=[
#             {'role': 'user', 'content': '测试'}
#         ],
#         max_tokens=1,
#     )
#     print("设置成功")
#     # print(response.choices[0].message.content)
# except Exception as e:
#     print(f"API 可能有问题，请检查：{e}")


def reset():
    """
    清空对话记录
    :return:
        List: 空的对话记录列表
    """
    return []


# def interact_summarization(prompt, article, temp=1.0):
#     """
#     调用参数生成摘要
#
#     :param prompt: (str)用于摘要提示词
#     :param article: (str)需要摘要的文章内容
#     :param temp: (float)模型温度，控制输出创造性（默认1.0）
#
#     :return: List[Tuple[str, str]]: 对话记录，包含输入文本与模型输出
#     """
#     # 合成请求文本
#     input_text = f"{prompt}\n{article}"
#
#     response = client.chat.completions.create(
#         model="qwen-plus-0806",
#         messages=[
#             {'role': 'system', 'content': '你是一个专业的高级文学学者，总是能精炼地将文章概括成几句话'},
#             {'role': 'user', 'content': input_text},
#         ],
#         temperature=temp,
#     )
#     return [(input_text, response.choices[0].message.content)]


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
