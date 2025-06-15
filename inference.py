import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import json

# === 参数配置 ===
model_dir = "./output/checkpoint-epoch10"  # 替换为你实际保存模型的路径
input_path = "./data/test1.json"
output_path = "test1_pred.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 当前使用设备: {device}")

# === 加载 tokenizer 和 模型 ===
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
model.eval()

# === 读取测试数据 ===
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 检查格式
assert isinstance(raw_data, list) and "content" in raw_data[0], "❌ 输入 JSON 格式不正确，缺少 'content' 字段"

# === 推理函数 ===
def predict(text):
    # 编码输入文本
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(device)

    # 使用 AMP 自动混合精度进行推理
    with torch.cuda.amp.autocast():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=4,  # Beam search 提高质量
            early_stopping=True,
            repetition_penalty=1.2  # 抑制重复
        )

    # 解码结果
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.strip()

# === 执行推理 ===
predictions = []
print("🔍 开始推理...")
for sample in tqdm(raw_data, desc="推理中"):
    content = sample["content"]
    result = predict(content)
    predictions.append(result)

# === 写入结果到文件 ===
with open(output_path, "w", encoding="utf-8") as f:
    for line in predictions:
        f.write(line + "\n")

print(f"✅ 所有样本推理完成，结果已保存至：{output_path}")
