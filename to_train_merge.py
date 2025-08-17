import json

def merge_consecutive(messages):
    merged = []
    buffer = []
    current_role = None

    for msg in messages:
        role, content = msg["role"], msg["content"]

        if role == current_role:
            buffer.append(content)
        else:
            if buffer:
                merged.append({
                    "role": current_role,
                    "content": "\n\n".join(buffer)
                })
            current_role = role
            buffer = [content]

    # 收尾
    if buffer:
        merged.append({
            "role": current_role,
            "content": "\n\n".join(buffer)
        })
    return merged


def process_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf8") as fin, \
         open(output_file, "w", encoding="utf8") as fout:
        for line in fin:
            data = json.loads(line)
            data["messages"] = merge_consecutive(data["messages"])
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    process_jsonl("train_200.jsonl", "train_merged_200.jsonl")
    print("✅ 已生成 train_merged.jsonl")
