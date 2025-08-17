from wxauto import WeChat
from wxauto.msgs import FriendMessage
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel



# ===== 路径配置 =====
MERGED_MODEL = r".\WeChat-DialogLM"

# ===== 加载分词器 =====
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# ===== 加载模型（合并后的就是完整模型） =====
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# ===== 聊天函数 =====
def chat_model(messages, max_new_tokens=256):
    """
    messages: [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}]
    """
    # 拼接成 Qwen 格式
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"  # 让模型继续生成助手的回答

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.9
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

# ===== 测试对话 =====
history = [{
  "role": "system",
  "content": "你是xxxxxxxxxxxxxxxxxxx。你正在和xxxxxxxxxx聊天，请完全延续你们之前的对话风格、语气和习惯用词。回答时保持自然、亲切、有生活气息，就像在微信上和对方随意交流一样，可以使用简短的句子、表情符号、语气词等，避免生硬的书面语。尽量模仿之前的聊天节奏与内容衔接。"
}]


wx = WeChat()

# 消息处理函数
def on_message(msg, chat):
    query = f'[{msg.type} {msg.attr}]{chat} - {msg.content}'
    print(query)
    if msg.content == "以下为新消息":
        return
    history.append({"role": "user", "content": msg.content})
    answer = chat_model(history)
    history.append({"role": "assistant", "content": answer})
    

    if isinstance(msg, FriendMessage):
        print(answer)
        for i in answer.split("\n\n"):
            msg.reply(i)
            time.sleep(1)


# 添加监听，监听到的消息用on_message函数进行处理
wx.AddListenChat(nickname="阿柒", callback=on_message)

time.sleep(6000)