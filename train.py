import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# === 参数设置 ===
model_name_or_path = "./model"  # 请确保路径下有 config.json、pytorch_model.bin、tokenizer
batch_size = 4
num_epochs = 10
max_input_length = 128
max_target_length = 256
save_path = "./output"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 当前设备：{device}")

# === 加载模型和 tokenizer ===
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)

# === 加载数据集 ===
dataset = load_from_disk("preprocessed_data")

def preprocess_function(examples):
    inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    targets = tokenizer(
        examples["target"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)
train_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

# === 优化器与 scheduler ===
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# === 混合精度训练相关 ===
scaler = torch.cuda.amp.GradScaler()

# === 创建输出目录 ===
os.makedirs(save_path, exist_ok=True)

# === 评估函数 ===
def compute_metrics(model, tokenizer, dataloader, device):
    model.eval()
    match_count = 0
    total_count = 0
    total_precision = []

    for batch in tqdm(dataloader, desc="🔍 正在评估"):
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=4
            )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, target in zip(preds, targets):
            pred = pred.strip()
            target = target.strip()

            total_count += 1
            if pred == target:
                match_count += 1

            pred_set = set([x.strip() for x in pred.split("[SEP]") if x.strip()])
            target_set = set([x.strip() for x in target.split("[SEP]") if x.strip()])

            if len(pred_set) == 0:
                total_precision.append(0)
            else:
                correct = len(pred_set & target_set)
                precision = correct / len(pred_set)
                total_precision.append(precision)

    accuracy = match_count / total_count
    precision = np.mean(total_precision)
    return accuracy, precision

# === 开始训练 ===
for epoch in range(num_epochs):
    print(f"\n🌀 Epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0.0

    train_iter = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
    for step, batch in enumerate(train_iter):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        train_iter.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    print(f"✅ Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")

    # === 评估并输出准确率和精确率 ===
    acc, pre = compute_metrics(model, tokenizer, train_dataloader, device)
    print(f"📊 Epoch {epoch + 1} 准确率 Acc: {acc:.4f}，精确率 Pre: {pre:.4f}")

    # === 保存模型 ===
    save_dir = os.path.join(save_path, f"checkpoint-epoch{epoch + 1}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"💾 模型保存至：{save_dir}")
