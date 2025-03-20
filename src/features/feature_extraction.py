import pandas as pd
import numpy as np
import os
import json
import pickle
import gensim
from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# 定义路径
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features')
os.makedirs(FEATURES_DIR, exist_ok=True)

# 加载词汇表
def load_vocabulary():
    """加载预先构建的词汇表"""
    vocab_path = os.path.join(PROCESSED_DATA_DIR, 'vocabulary.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)
        return vocabulary
    else:
        return None

# LSA特征提取（潜在语义分析）
def extract_lsa_features(train_df, val_df=None, test_df=None, n_components=100, max_features=5000, vocabulary=None):
    """使用LSA（潜在语义分析）提取文本特征"""
    # 如果提供了词汇表，使用它；否则，从训练数据构建
    if vocabulary is None:
        vocabulary = load_vocabulary()
    
    # 创建LSA管道：TF-IDF向量化 + 截断SVD
    lsa_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, vocabulary=vocabulary)),
        ('svd', TruncatedSVD(n_components=n_components, random_state=42)),
        ('scaler', StandardScaler())
    ])
    
    # 转换训练集
    X_train = lsa_pipeline.fit_transform(train_df['processed_text'])
    
    # 转换验证集和测试集（如果提供）
    X_val = lsa_pipeline.transform(val_df['processed_text']) if val_df is not None else None
    X_test = lsa_pipeline.transform(test_df['processed_text']) if test_df is not None else None
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['intent'])
    y_val = label_encoder.transform(val_df['intent']) if val_df is not None else None
    y_test = label_encoder.transform(test_df['intent']) if test_df is not None else None
    
    # 保存特征和标签
    np.save(os.path.join(FEATURES_DIR, 'X_train_lsa.npy'), X_train)
    
    if X_val is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_val_lsa.npy'), X_val)
    
    if X_test is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_test_lsa.npy'), X_test)
    
    # 保存管道
    with open(os.path.join(FEATURES_DIR, 'lsa_pipeline.pkl'), 'wb') as f:
        pickle.dump(lsa_pipeline, f)
    
    print(f"LSA特征提取完成")
    print(f"训练集特征形状: {X_train.shape}")
    if X_val is not None:
        print(f"验证集特征形状: {X_val.shape}")
    if X_test is not None:
        print(f"测试集特征形状: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, lsa_pipeline

# TF-IDF特征提取
def extract_tfidf_features(train_df, val_df=None, test_df=None, max_features=5000, vocabulary=None):
    """使用TF-IDF提取文本特征"""
    # 如果提供了词汇表，使用它；否则，从训练数据构建
    if vocabulary is None:
        vocabulary = load_vocabulary()
    
    # 创建TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, vocabulary=vocabulary)
    
    # 转换训练集
    X_train = tfidf_vectorizer.fit_transform(train_df['processed_text'])
    
    # 转换验证集和测试集（如果提供）
    X_val = tfidf_vectorizer.transform(val_df['processed_text']) if val_df is not None else None
    X_test = tfidf_vectorizer.transform(test_df['processed_text']) if test_df is not None else None
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['intent'])
    y_val = label_encoder.transform(val_df['intent']) if val_df is not None else None
    y_test = label_encoder.transform(test_df['intent']) if test_df is not None else None
    
    # 保存特征和标签
    np.save(os.path.join(FEATURES_DIR, 'X_train_tfidf.npy'), X_train.toarray())
    np.save(os.path.join(FEATURES_DIR, 'y_train.npy'), y_train)
    
    if X_val is not None and y_val is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_val_tfidf.npy'), X_val.toarray())
        np.save(os.path.join(FEATURES_DIR, 'y_val.npy'), y_val)
    
    if X_test is not None and y_test is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_test_tfidf.npy'), X_test.toarray())
        np.save(os.path.join(FEATURES_DIR, 'y_test.npy'), y_test)
    
    # 保存向量化器和标签编码器
    with open(os.path.join(FEATURES_DIR, 'tfidf_vectorizer.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'vocabulary': tfidf_vectorizer.vocabulary_,
            'idf': tfidf_vectorizer.idf_.tolist() if hasattr(tfidf_vectorizer, 'idf_') else None
        }, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(FEATURES_DIR, 'label_encoder.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'classes': label_encoder.classes_.tolist()
        }, f, ensure_ascii=False, indent=2)
    
    print(f"TF-IDF特征提取完成")
    print(f"训练集特征形状: {X_train.shape}")
    if X_val is not None:
        print(f"验证集特征形状: {X_val.shape}")
    if X_test is not None:
        print(f"测试集特征形状: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, tfidf_vectorizer, label_encoder

# 词袋模型特征提取
def extract_bow_features(train_df, val_df=None, test_df=None, max_features=5000, vocabulary=None):
    """使用词袋模型提取文本特征"""
    # 如果提供了词汇表，使用它；否则，从训练数据构建
    if vocabulary is None:
        vocabulary = load_vocabulary()
    
    # 创建词袋模型向量化器
    bow_vectorizer = CountVectorizer(max_features=max_features, vocabulary=vocabulary)
    
    # 转换训练集
    X_train = bow_vectorizer.fit_transform(train_df['processed_text'])
    
    # 转换验证集和测试集（如果提供）
    X_val = bow_vectorizer.transform(val_df['processed_text']) if val_df is not None else None
    X_test = bow_vectorizer.transform(test_df['processed_text']) if test_df is not None else None
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['intent'])
    y_val = label_encoder.transform(val_df['intent']) if val_df is not None else None
    y_test = label_encoder.transform(test_df['intent']) if test_df is not None else None
    
    # 保存特征和标签
    np.save(os.path.join(FEATURES_DIR, 'X_train_bow.npy'), X_train.toarray())
    
    if X_val is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_val_bow.npy'), X_val.toarray())
    
    if X_test is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_test_bow.npy'), X_test.toarray())
    
    # 保存向量化器
    with open(os.path.join(FEATURES_DIR, 'bow_vectorizer.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'vocabulary': bow_vectorizer.vocabulary_
        }, f, ensure_ascii=False, indent=2)
    
    print(f"词袋模型特征提取完成")
    print(f"训练集特征形状: {X_train.shape}")
    if X_val is not None:
        print(f"验证集特征形状: {X_val.shape}")
    if X_test is not None:
        print(f"测试集特征形状: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, bow_vectorizer

# 哈希特征提取
def extract_hash_features(train_df, val_df=None, test_df=None, n_features=5000):
    """使用哈希向量化提取文本特征，适用于大规模词汇表"""
    # 创建哈希向量化器
    hash_vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
    
    # 转换训练集
    X_train = hash_vectorizer.fit_transform(train_df['processed_text'])
    
    # 转换验证集和测试集（如果提供）
    X_val = hash_vectorizer.transform(val_df['processed_text']) if val_df is not None else None
    X_test = hash_vectorizer.transform(test_df['processed_text']) if test_df is not None else None
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['intent'])
    y_val = label_encoder.transform(val_df['intent']) if val_df is not None else None
    y_test = label_encoder.transform(test_df['intent']) if test_df is not None else None
    
    # 保存特征和标签
    np.save(os.path.join(FEATURES_DIR, 'X_train_hash.npy'), X_train.toarray())
    
    if X_val is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_val_hash.npy'), X_val.toarray())
    
    if X_test is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_test_hash.npy'), X_test.toarray())
    
    # 保存向量化器
    with open(os.path.join(FEATURES_DIR, 'hash_vectorizer.pkl'), 'wb') as f:
        pickle.dump(hash_vectorizer, f)
    
    print(f"哈希特征提取完成")
    print(f"训练集特征形状: {X_train.shape}")
    if X_val is not None:
        print(f"验证集特征形状: {X_val.shape}")
    if X_test is not None:
        print(f"测试集特征形状: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, hash_vectorizer

# Word2Vec特征提取
def extract_word2vec_features(train_df, val_df=None, test_df=None, vector_size=100, window=5, min_count=1):
    """使用Word2Vec提取文本特征，捕捉词语的语义关系"""
    # 准备训练数据
    train_texts = [text.split() for text in train_df['processed_text']]
    
    # 训练Word2Vec模型
    w2v_model = Word2Vec(sentences=train_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    
    # 定义文档向量化函数（使用词向量平均值表示文档）
    def document_vector(doc, model):
        doc_words = doc.split()
        word_vectors = [model.wv[word] for word in doc_words if word in model.wv]
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)
    
    # 转换训练集
    X_train = np.array([document_vector(doc, w2v_model) for doc in train_df['processed_text']])
    
    # 转换验证集和测试集（如果提供）
    X_val = np.array([document_vector(doc, w2v_model) for doc in val_df['processed_text']]) if val_df is not None else None
    X_test = np.array([document_vector(doc, w2v_model) for doc in test_df['processed_text']]) if test_df is not None else None
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['intent'])
    y_val = label_encoder.transform(val_df['intent']) if val_df is not None else None
    y_test = label_encoder.transform(test_df['intent']) if test_df is not None else None
    
    # 保存特征和标签
    np.save(os.path.join(FEATURES_DIR, 'X_train_w2v.npy'), X_train)
    
    if X_val is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_val_w2v.npy'), X_val)
    
    if X_test is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_test_w2v.npy'), X_test)
    
    # 保存模型
    w2v_model.save(os.path.join(FEATURES_DIR, 'word2vec_model.model'))
    
    print(f"Word2Vec特征提取完成")
    print(f"训练集特征形状: {X_train.shape}")
    if X_val is not None:
        print(f"验证集特征形状: {X_val.shape}")
    if X_test is not None:
        print(f"测试集特征形状: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, w2v_model

# FastText特征提取
def extract_fasttext_features(train_df, val_df=None, test_df=None, vector_size=100, window=5, min_count=1):
    """使用FastText提取文本特征，能更好地处理未登录词和词形变化"""
    # 准备训练数据
    train_texts = [text.split() for text in train_df['processed_text']]
    
    # 训练FastText模型
    ft_model = FastText(sentences=train_texts, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    
    # 定义文档向量化函数（使用词向量平均值表示文档）
    def document_vector(doc, model):
        doc_words = doc.split()
        word_vectors = [model.wv[word] for word in doc_words]
        return np.mean(word_vectors, axis=0) if len(word_vectors) > 0 else np.zeros(model.vector_size)
    
    # 转换训练集
    X_train = np.array([document_vector(doc, ft_model) for doc in train_df['processed_text']])
    
    # 转换验证集和测试集（如果提供）
    X_val = np.array([document_vector(doc, ft_model) for doc in val_df['processed_text']]) if val_df is not None else None
    X_test = np.array([document_vector(doc, ft_model) for doc in test_df['processed_text']]) if test_df is not None else None
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['intent'])
    y_val = label_encoder.transform(val_df['intent']) if val_df is not None else None
    y_test = label_encoder.transform(test_df['intent']) if test_df is not None else None
    
    # 保存特征和标签
    np.save(os.path.join(FEATURES_DIR, 'X_train_fasttext.npy'), X_train)
    
    if X_val is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_val_fasttext.npy'), X_val)
    
    if X_test is not None:
        np.save(os.path.join(FEATURES_DIR, 'X_test_fasttext.npy'), X_test)
    
    # 保存模型
    ft_model.save(os.path.join(FEATURES_DIR, 'fasttext_model.model'))
    
    print(f"FastText特征提取完成")
    print(f"训练集特征形状: {X_train.shape}")
    if X_val is not None:
        print(f"验证集特征形状: {X_val.shape}")
    if X_test is not None:
        print(f"测试集特征形状: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, ft_model