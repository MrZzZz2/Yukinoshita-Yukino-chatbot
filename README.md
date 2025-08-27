
# 雪之下雪乃 角色扮演聊天机器人

一个基于 [Qwen 模型](https://modelscope.cn/organization/Qwen?tab=all) 的角色扮演聊天机器人。  
在 Gradio 界面中，你可以和《我的青春恋爱物语果然有问题》里的 **雪之下雪乃** 进行沉浸式对话，支持记忆提取与保存。

## ✨ 功能
- 多轮对话，保留上下文
- 长期记忆提取（存储在 `files/memory.json`）
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
- 更改你喜欢的角色
- 优化记忆提取逻辑
- 加入 Docker 支持