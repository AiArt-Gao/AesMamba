import pandas as pd

# 读取CSV文件
df = pd.read_csv('/data/sjq/IAA_Aesmamba/EXPV2/PARA_PIAA/aesthetic_metric.csv')

# 计算plcc和srcc列的均值
plcc_mean = df['plcc'].mean()
srcc_mean = df['srcc'].mean()

print(f'PLCC Mean: {plcc_mean}')
print(f'SRCC Mean: {srcc_mean}')
