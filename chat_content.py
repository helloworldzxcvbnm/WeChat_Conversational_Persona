import os
import sqlite3
import json

# 配置部分
DB_DIR = r"./db_files"   # 存放多个db文件的目录
OUTPUT_JSON = "messages.json"
TARGET_TALKER = "wxid_k7ydmy177nxxxx"  # 目标 StrTalker

def extract_messages_from_db(db_path):
    """从单个SQLite DB文件中提取消息"""
    messages = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT CreateTime, MsgSequence, StrContent
            FROM MSG
            WHERE StrTalker = ?
        """, (TARGET_TALKER,))
        
        for create_time, msg_sequence, str_content in cursor.fetchall():
            if not str_content or "<msg>" in str_content:
                continue
            messages.append({
                "CreateTime": create_time,
                "MsgSequence": msg_sequence,
                "StrContent": str_content
            })

        conn.close()
    except Exception as e:
        print(f"[ERROR] 处理文件 {db_path} 出错: {e}")
    return messages


def main():
    all_messages = []
    for file_name in os.listdir(DB_DIR):
        if file_name.endswith(".db"):
            db_path = os.path.join(DB_DIR, file_name)
            print(f"正在处理: {db_path}")
            msgs = extract_messages_from_db(db_path)
            all_messages.extend(msgs)

    # 按 CreateTime 降序排列
    all_messages.sort(key=lambda x: x["CreateTime"], reverse=True)

    # 添加自增 id
    for idx, msg in enumerate(all_messages, start=1):
        msg["id"] = idx

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，总记录数 {len(all_messages)}，结果已保存到 {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
