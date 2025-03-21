import pandas as pd
import os
from feature_extraction import extract_tfidf_features

# 获取当前脚本的目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 加载数据
train_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_processed.csv'))
val_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'val_processed.csv'))
test_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_processed.csv'))

# 提取TF-IDF特征
print('开始提取TF-IDF特征...')
extract_tfidf_features(train_df, val_df, test_df)
print('特征提取完成！')