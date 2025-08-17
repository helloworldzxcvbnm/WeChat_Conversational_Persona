import json

# 输入文件和输出文件
input_file = "messages.json"
output_file = "train_200.jsonl"

# 读取聊天记录
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 按时间排序
data.sort(key=lambda x: x["CreateTime"])

sessions = []
current_session = []
last_time = None

for msg in data:
    t = msg["CreateTime"]
    content = msg["StrContent"].strip()

    if not content:  # 跳过空消息
        continue

    # 新会话判断
    if last_time is None:
        current_session.append(msg)
    else:
        if t - last_time > 200:  # 超过5分钟，新会话
            if current_session:
                sessions.append(current_session)
            current_session = [msg]
        else:
            current_session.append(msg)
    last_time = t

# 最后一组加入
if current_session:
    sessions.append(current_session)

# 转换成微调数据（messages格式）
train_data = []
id_counter = 1

for session in sessions:
    messages = []
    for m in session:
        role = "user" if m["MsgSequence"] == 0 else "assistant"
        messages.append({"role": role, "content": m["StrContent"].strip()})
    train_data.append({"id": id_counter, "messages": messages})
    id_counter += 1

# 保存为 JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"处理完成！共 {len(train_data)} 轮对话，已保存到 {output_file}")
