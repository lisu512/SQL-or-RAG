# SQL或RAG意图分类模型

这个项目旨在训练一个高精度模型，用于预测用户问题的意图是想查询知识库（RAG）还是想进行SQL查询。这是完成大模型搜索框功能的一个关键步骤，能够帮助系统自动选择合适的查询方式。

## 项目结构

```
├── data/                  # 数据集目录
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
│       └── features/      # 提取的特征
├── models/                # 保存训练好的模型
├── notebooks/            # Jupyter notebooks用于探索性分析
├── src/                   # 源代码
│   ├── data/              # 数据处理相关代码
│   │   ├── generate_dataset.py  # 生成训练数据
│   │   └── preprocess.py        # 数据预处理
│   ├── features/          # 特征工程相关代码
│   │   └── feature_extraction.py # 特征提取
│   ├── models/            # 模型训练和评估相关代码
│   │   └── train_model.py        # 模型训练和评估
│   └── utils/             # 工具函数
│       └── model_utils.py        # 模型评估和特征选择工具
└── requirements.txt       # 项目依赖
```

## 功能描述

该模型能够准确区分两种类型的用户查询意图：

1. **SQL查询意图**：用户想要查询结构化数据，例如「公司男女比例是多少」、「销售部门的平均工资是多少」、「查询研发部门2023年的人员构成」等。

2. **知识库查询意图（RAG）**：用户想要查询非结构化知识，例如「公司人工客服的电话号码是多少」、「公司的休假政策是什么」、「如何申请公司提供的教育补贴」等。

## 特点

- **超大规模数据集**：包含6000+样本，覆盖多种表达方式和查询场景，支持多样化的口语表达
- **高级特征提取**：支持TF-IDF、词袋模型、LSA、哈希特征、Word2Vec和FastText等多种特征提取方法
- **多模型支持**：实现了逻辑回归、SVM、随机森林、梯度提升树、朴素贝叶斯等多种基础模型
- **高级集成学习**：通过投票集成、堆叠集成、AdaBoost和Bagging等方法显著提高预测准确率
- **增强数据增强**：使用多种文本增强技术扩充训练数据，包括词序调整、前缀添加、同义词替换等
- **特征选择**：支持基于卡方统计量、互信息和基于树的特征选择方法
- **完整评估**：提供准确率、精确率、召回率、F1值、ROC曲线、学习曲线等多维度评估指标
- **模型可视化**：提供混淆矩阵、学习曲线、验证曲线等可视化工具，帮助理解模型性能

## 使用方法

### 1. 生成数据集

```bash
python src/data/generate_dataset.py
```

### 2. 数据预处理

```bash
python src/data/preprocess.py
```

### 3. 特征提取

```bash
python src/features/feature_extraction.py
```

### 4. 模型训练与评估

```bash
python src/models/train_model.py
```

### 5. 使用训练好的模型进行预测

```python
import json
import pickle
import numpy as np
from src.utils.model_utils import predict_intent

# 方法1：使用工具函数直接预测
query = "公司的休假政策是什么"
intent, probability = predict_intent(query, model_name='stacking', vectorizer_type='tfidf')
print(f"预测结果: {intent}, 概率: {probability:.4f}")

# 方法2：手动加载模型和特征提取器
with open('models/stacking.pkl', 'rb') as f:
    model = pickle.load(f)

# 加载特征提取器
with open('data/processed/features/tfidf_vectorizer.json', 'r') as f:
    vectorizer_data = json.load(f)
    
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=vectorizer_data['vocabulary'])
if 'idf' in vectorizer_data and vectorizer_data['idf'] is not None:
    vectorizer.idf_ = np.array(vectorizer_data['idf'])

# 预处理文本
from src.data.preprocess import clean_text, tokenize
processed_text = tokenize(clean_text("你的查询文本"))

# 提取特征
features = vectorizer.transform([processed_text])

# 预测
prediction = model.predict(features)
print("预测结果:", "SQL查询" if prediction[0] == 0 else "知识库查询")
```

## 高级用法

### 使用Word2Vec特征

```python
from src.features.feature_extraction import extract_word2vec_features

# 加载数据
train_df = pd.read_csv('data/processed/train_processed.csv')
val_df = pd.read_csv('data/processed/val_processed.csv')

# 提取Word2Vec特征
X_train, y_train, X_val, y_val, _, _, w2v_model = extract_word2vec_features(train_df, val_df)
```

### 使用特征选择

```python
from src.utils.model_utils import select_features

# 选择最重要的特征
X_train_selected, X_val_selected, X_test_selected, selector = select_features(
    X_train, y_train, X_val, X_test, method='chi2', k=500
)
```

### 模型评估与可视化

```python
from src.utils.model_utils import evaluate_model_detailed, plot_learning_curve, plot_roc_curve

# 详细评估模型
accuracy, precision, recall, f1, cm, report = evaluate_model_detailed(
    model, X_test, y_test, class_names=['sql', 'rag']
)

# 绘制学习曲线
plot_learning_curve(model, X_train, y_train).savefig('learning_curve.png')

# 绘制ROC曲线
plot_roc_curve(model, X_test, y_test, class_names=['sql', 'rag']).savefig('roc_curve.png')
```