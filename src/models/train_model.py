import os
import numpy as np
import json
import pickle
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 定义路径
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 加载特征数据
def load_features(feature_type='tfidf'):
    """加载特征数据"""
    X_train = np.load(os.path.join(FEATURES_DIR, f'X_train_{feature_type}.npy'))
    y_train = np.load(os.path.join(FEATURES_DIR, 'y_train.npy'))
    
    # 检查是否有验证集和测试集
    X_val, y_val = None, None
    X_test, y_test = None, None
    
    if os.path.exists(os.path.join(FEATURES_DIR, f'X_val_{feature_type}.npy')):
        X_val = np.load(os.path.join(FEATURES_DIR, f'X_val_{feature_type}.npy'))
        y_val = np.load(os.path.join(FEATURES_DIR, 'y_val.npy'))
    
    if os.path.exists(os.path.join(FEATURES_DIR, f'X_test_{feature_type}.npy')):
        X_test = np.load(os.path.join(FEATURES_DIR, f'X_test_{feature_type}.npy'))
        y_test = np.load(os.path.join(FEATURES_DIR, 'y_test.npy'))
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 评估模型在测试集上的性能
def evaluate_model(model, X_test, y_test, label_encoder):
    """评估模型在测试集上的性能"""
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'confusion_matrix_stacking.png'))
    
    print(f"测试集准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(report)
    
    return accuracy, report, cm

# 保存模型
def save_model(model, model_name):
    """保存模型到文件"""
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {model_path}")

# 训练堆叠集成模型
def train_stacking_ensemble(X_train, y_train, X_val=None, y_val=None):
    """训练堆叠集成模型，使用元学习器组合多个基础模型的预测"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 创建并优化基础模型，使用更快的配置
    estimators = [
        ('lr', LogisticRegression(max_iter=100, tol=1e-2, C=1.0, random_state=42)),
        ('svm', SVC(probability=True, kernel='linear', C=1.0, tol=1e-2, random_state=42, cache_size=1000)),
        ('rf', RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1))
    ]
    
    # 创建堆叠集成模型，使用逻辑回归作为元学习器，减少交叉验证折数
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=100, tol=1e-2, random_state=42),
        cv=cv,
        n_jobs=-1
    )
    
    # 定义参数分布（用于随机搜索）
    param_distributions = {
        'lr__C': [1.0],
        'svm__C': [1.0],
        'rf__n_estimators': [30],
        'final_estimator__C': [1.0]
    }
    
    # 使用随机搜索找到最佳参数
    print("\n开始训练堆叠集成模型...")
    print("使用以下配置:")
    print("- 基础模型: 逻辑回归、支持向量机(线性核)、随机森林(30棵树)")
    print("- 交叉验证: 2折")
    print("- 随机搜索: 3次迭代")
    
    # 创建带有详细日志的随机搜索
    random_search = RandomizedSearchCV(
        stacking,
        param_distributions,
        n_iter=3,  # 减少搜索次数
        cv=2,     # 减少交叉验证折数
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # 训练模型
    start_time = time.time()
    with tqdm(total=6, desc="训练进度") as pbar:  # 2折 * 3次迭代
        def update_progress(*args):
            pbar.update(1)
        random_search.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"\n训练完成！总耗时: {train_time:.2f}秒")
    print("最佳参数组合:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"最佳交叉验证得分: {random_search.best_score_:.4f}")
    
    # 输出所有参数组合的结果
    print("\n所有参数组合的得分:")
    means = random_search.cv_results_['mean_test_score']
    stds = random_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, random_search.cv_results_['params']):
        print(f"  参数: {params}")
        print(f"  得分: {mean:.4f} (+/- {std * 2:.4f})\n")
    
    # 获取最佳模型
    best_model = random_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"堆叠集成模型训练时间: {train_time:.2f}秒")
    print(f"最佳参数: {random_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

# 主函数
def main(feature_type='tfidf'):
    """主函数，训练和评估堆叠集成模型"""
    # 加载特征数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_features(feature_type)
    
    # 加载标签编码器
    with open(os.path.join(FEATURES_DIR, 'label_encoder.json'), 'r', encoding='utf-8') as f:
        label_encoder_data = json.load(f)
    
    # 创建标签编码器对象
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_data['classes'])
    
    # 训练堆叠集成模型
    print("\n训练堆叠集成模型...")
    stacking_model = train_stacking_ensemble(X_train, y_train, X_val, y_val)
    
    # 在测试集上评估模型
    if X_test is not None and y_test is not None:
        print("\n在测试集上评估堆叠集成模型:")
        evaluate_model(stacking_model, X_test, y_test, label_encoder)
    
    # 保存模型
    save_model(stacking_model, 'stacking_ensemble')

if __name__ == "__main__":
    main(feature_type='tfidf')