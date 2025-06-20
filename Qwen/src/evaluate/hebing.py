import json
import pandas as pd

# 1. 读取三个JSON文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

# 加载各文件数据
overall_data = load_json('/home/lrz/Qwen/output/data/overall.json')
sharpness_data = load_json('/home/lrz/Qwen/output/data/sharpness.json')
fidelity_data = load_json('/home/lrz/Qwen/output/data/fidelity.json')

# 2. 转换为DataFrame并统一格式
def to_df(data, metric_name):
    df = pd.DataFrame(data)
    df.rename(columns={'score': metric_name}, inplace=True)  # 假设每个文件中的评分字段都叫'score'
    return df[['image', metric_name]]  # 只保留图片名和评分

df_overall = to_df(overall_data, 'overall')
df_sharpness = to_df(sharpness_data, 'sharpness')
df_fidelity = to_df(fidelity_data, 'fidelity')

# 3. 按图片名合并三个DataFrame
merged_df = df_overall.merge(df_sharpness, on='image').merge(df_fidelity, on='image')
merged_df.rename(columns={'image': 'filename'}, inplace=True)
# 4. 保存为CSV
merged_df.to_csv('/home/lrz/Qwen/output/data/image_quality_metrics_1.csv', index=False)

print(f"合并完成，共处理{len(merged_df)}张图片")
print("生成的CSV文件包含以下列:", merged_df.columns.tolist())