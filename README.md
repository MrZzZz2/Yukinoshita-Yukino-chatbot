# 雪之下雪乃 角色扮演聊天机器人

一个基于 [Qwen 模型](https://modelscope.cn/organization/Qwen?tab=all) 的角色扮演聊天机器人。  
在 Gradio 界面中，你可以和《我的青春恋爱物语果然有问题》里的 **雪之下雪乃** 进行沉浸式对话，支持记忆提取与保存。

## ✨ 功能
- 多轮对话，使用滑动窗口机制保留最近x条上下文
- 长期记忆提取，每次对话完使用语言模型进行对话总结，存储在 `files/memory.json`
- 记忆遗忘，使用滑动窗口机制，只记忆最近总结的x条记忆
- 角色扮演提示词，逼真还原人物性格
- Gradio WebUI 界面

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/yourname/your-project.git
````

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行项目
```bash
python chatbot.py
```

然后在浏览器中访问 [http://127.0.0.1:7860](http://127.0.0.1:7860/) （不一定是这个网址，以实际为准）就能开始对话。

## 📂 文件说明

- `main.py`：核心脚本
- `files/memory.json`：对话记忆存储
- `requirements.txt`：依赖文件
- `.gitignore`：忽略上传的文件
- `LICENSE`：开源许可证

## 📝 TODO

- 你可以修改prompt，自定义你喜欢的角色进行对话
- 现在实现的功能都很基本，你可以添加的功能：
  - 使用AI决定什么时候该储存记忆
  - 决定存储新记忆时和旧记忆使什么方法融合（直接使用一个语言模型（针对这个功能做STF或RLHF后的模型效果可能会更好）下prompt，定时进行记忆压缩、总结）
  - 使用AI决定该忘记什么样的记忆（直接使用一个语言模型（针对这个功能做STF或RLHF后的模型效果可能会更好）下prompt，定时忘记一部分不重要的记忆）
  - 我记得李宏毅老师好像有个功能可以加速语言模型的对话生成速度，详见[bilibili视频](https://www.bilibili.com/video/BV1BJ4m1e7g8?spm_id_from=333.788.videopod.episodes&vd_source=108c7957d88a7eafb172b276df7068cf&p=33)或[youtube视频](https://www.youtube.com/watch?v=MAbGgsWKrg8&list=PLJV_el3uVTsPz6CTopeRp2L2t4aL_KgiI&index=17)
