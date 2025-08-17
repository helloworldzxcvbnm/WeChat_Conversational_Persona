import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch
import swanlab


# ========== 配置 ==========
MODEL_NAME = r"F:\python_project\big_model\Qwen2.5-1.5B-Instruct"
TRAIN_FILE = "train_merged_200.jsonl"
OUTPUT_DIR = "./lora-qwen2.5-chat-v2.0"
BATCH_SIZE = 1
MAX_SEQ_LEN = 1024
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# ========== 初始化 SwanLab ==========
swanlab.init(
    project="wechat-chat-lora",
    experiment_name="qwen2.5-lora-finetune-v2.0",
    config={
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "lora_r": LORA_R
    }
)

# ========== 数据集加载 & 切分 ==========
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]

# 格式化对话为纯文本
def format_chat(example):
    text = ""
    for m in example["messages"]:
        if m["role"] == "user":
            text += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        else:
            text += f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"
    return {"text": text}

train_ds = train_ds.map(format_chat)
val_ds = val_ds.map(format_chat)

# 分词器 (适配Qwen2.5格式)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 显式添加pad token

def tokenize_fn(examples):
    tokenized = tokenizer(
        examples["text"],
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length"
    )
    # 创建labels - 因果语言建模需要labels与input_ids相同
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

# ========== 模型加载 + LoRA ==========
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# 适配Qwen2.5的LoRA目标模块
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印可训练参数信息

# ========== 训练参数 ==========
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=2,
    logging_steps=10,
    save_steps=100,
    # 修复：将 evaluation_strategy 改为 eval_strategy
    eval_strategy="steps",  # 定期评估
    eval_steps=50,
    report_to=["swanlab"],
    optim="paged_adamw_8bit",  # 更稳定的优化器
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

# 使用更适合因果语言建模的Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言建模
)

# ========== 自定义 Trainer（接入 SwanLab） ==========
class SwanLabTrainer(Trainer):
    def log(self, logs, *args, **kwargs):  # 捕获额外参数
        super().log(logs)
        swanlab.log(logs)  # 同步到 SwanLab


trainer = SwanLabTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ========== 开始训练 ==========
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"微调完成，模型已保存到 {OUTPUT_DIR}")