import pandas as pd
from feature_extraction import extract_tfidf_features

# 加载数据
train_df = pd.read_csv('../../data/processed/train_processed.csv')
val_df = pd.read_csv('../../data/processed/val_processed.csv')
test_df = pd.read_csv('../../data/processed/test_processed.csv')

# 提取TF-IDF特征
print('开始提取TF-IDF特征...')
extract_tfidf_features(train_df, val_df, test_df)
print('特征提取完成！')