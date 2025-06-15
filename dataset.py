import json
from datasets import Dataset

def preprocess(file_path: str, save_path: str = "preprocessed_data"):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data_list = []
    skipped = 0

    for item in raw_data:
        content = item.get("content", "").strip()
        output = item.get("output", "").strip()

        # 跳过空数据
        if not content or not output:
            skipped += 1
            continue

        # 检查输出是否符合格式要求
        if "[END]" not in output:
            print(f"⚠️ 样本ID {item.get('id', 'N/A')} 缺少 [END]：{output}")
            skipped += 1
            continue

        data_list.append({
            "input": content,
            "target": output
        })

    dataset = Dataset.from_list(data_list)
    dataset.save_to_disk(save_path)
    print(f"✅ 成功处理 {len(data_list)} 条数据，跳过 {skipped} 条异常数据，已保存至：{save_path}")

if __name__ == "__main__":
    preprocess("./data/train.json")
