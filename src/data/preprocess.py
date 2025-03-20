import pandas as pd
import re
import os
import jieba
import json
import numpy as np
from sklearn.utils import shuffle

# 定义路径
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 常用中文停用词列表 - 扩充
STOP_WORDS = set([
    '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '有', '在', '中', '为', 
    '对', '或', '以', '于', '上', '下', '之', '由', '等', '被', '一', '将', '从', '但', '向', '到', 
    '它', '们', '我', '你', '他', '她', '我们', '你们', '他们', '她们', '这个', '那个', '这些', '那些',
    '啊', '呀', '哦', '哎', '嗯', '呢', '吧', '啦', '么', '吗', '呵', '嘛', '哈', '哟', '喂', '哩',
    '如何', '怎么', '怎样', '什么', '哪些', '为什么', '怎么样', '多少', '几个', '如果', '可以', '能否',
    '请问', '麻烦', '帮忙', '想要', '需要', '希望', '想', '要', '能', '可', '会', '应该', '应当', '得'
])

# 文本清洗
def clean_text(text):
    """清洗文本，去除特殊字符和多余空格"""
    # 去除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 分词
def tokenize(text, remove_stopwords=True):
    """使用jieba进行中文分词，可选是否去除停用词"""
    words = jieba.cut(text)
    if remove_stopwords:
        words = [word for word in words if word not in STOP_WORDS and len(word.strip()) > 0]
    return ' '.join(words)

# 文本增强
def augment_text(text, intent, augment_prob=0.5):
    """对文本进行多种数据增强，包括词序调整、前缀添加、同义词替换等"""
    # 增加增强概率，使数据集更加多样化
    if np.random.random() > augment_prob:
        return text
    
    words = text.split()
    
    # 对于较长的文本，可以进行词序调整
    if len(words) > 4 and np.random.random() < 0.5:
        # 随机选择一个位置，交换相邻的两个词
        idx = np.random.randint(0, len(words) - 1)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    
    # 随机删除一个非关键词（增强鲁棒性）
    if len(words) > 5 and np.random.random() < 0.3:
        # 避免删除第一个词，可能是关键动词
        idx = np.random.randint(1, len(words))
        words.pop(idx)
    
    # 对于SQL查询，可以添加一些常见的前缀
    if intent == 'sql' and np.random.random() < 0.4:
        prefixes = ['请', '帮我', '能否', '想知道', '请问', '麻烦', '帮忙', '希望', '需要', '想要']
        prefix = np.random.choice(prefixes)
        words = [prefix] + words
    
    # 对于RAG查询，可以添加一些常见的前缀
    if intent == 'rag' and np.random.random() < 0.4:
        prefixes = ['请问', '想了解', '告诉我', '我想知道', '能否介绍', '如何', '怎样才能', '有没有人知道']
        prefix = np.random.choice(prefixes)
        words = [prefix] + words
    
    # 添加一些常见的后缀
    if np.random.random() < 0.3:
        suffixes = ['呢', '啊', '吗', '呀', '了', '的', '？']
        suffix = np.random.choice(suffixes)
        if not words[-1].endswith(suffix):
            words[-1] = words[-1] + suffix
    
    return ' '.join(words)

# 构建词汇表
def build_vocabulary(texts, max_vocab_size=10000):
    """从文本列表中构建词汇表"""
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序并限制词汇表大小
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:max_vocab_size])}
    
    return vocabulary

# 预处理数据
def preprocess_data(train_path=None, val_path=None, test_path=None, build_vocab=True, max_vocab_size=15000, augment=True):
    """预处理训练集、验证集和测试集数据"""
    # 如果未提供路径，使用默认路径
    if train_path is None:
        train_path = os.path.join(RAW_DATA_DIR, 'train.csv')
    if val_path is None:
        val_path = os.path.join(RAW_DATA_DIR, 'val.csv')
    if test_path is None:
        test_path = os.path.join(RAW_DATA_DIR, 'test.csv')
    
    # 加载数据
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else None
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    # 清洗和分词
    train_df['processed_text'] = train_df['query'].apply(clean_text).apply(lambda x: tokenize(x, remove_stopwords=True))
    if val_df is not None:
        val_df['processed_text'] = val_df['query'].apply(clean_text).apply(lambda x: tokenize(x, remove_stopwords=True))
    if test_df is not None:
        test_df['processed_text'] = test_df['query'].apply(clean_text).apply(lambda x: tokenize(x, remove_stopwords=True))
    
    # 数据增强（仅对训练集）
    if augment:
        augmented_samples = []
        for _, row in train_df.iterrows():
            # 对原始文本进行增强
            augmented_text = augment_text(row['processed_text'], row['intent'])
            if augmented_text != row['processed_text']:
                new_row = row.copy()
                new_row['processed_text'] = augmented_text
                augmented_samples.append(new_row)
        
        # 如果有增强样本，添加到训练集
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            train_df = pd.concat([train_df, augmented_df], ignore_index=True)
            train_df = shuffle(train_df, random_state=42)  # 打乱数据顺序
            print(f"数据增强后的训练集大小: {len(train_df)}")
    
    # 构建词汇表
    if build_vocab:
        vocabulary = build_vocabulary(train_df['processed_text'], max_vocab_size)
        # 保存词汇表
        with open(os.path.join(PROCESSED_DATA_DIR, 'vocabulary.json'), 'w', encoding='utf-8') as f:
            json.dump(vocabulary, f, ensure_ascii=False, indent=2)
        print(f"词汇表构建完成，大小: {len(vocabulary)}")
    
    # 保存处理后的数据
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv'), index=False)
    if val_df is not None:
        val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val_processed.csv'), index=False)
    if test_df is not None:
        test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv'), index=False)
    
    print(f"数据预处理完成")
    print(f"训练集大小: {len(train_df)}")
    if val_df is not None:
        print(f"验证集大小: {len(val_df)}")
    if test_df is not None:
        print(f"测试集大小: {len(test_df)}")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = preprocess_data()