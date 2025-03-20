import os
import numpy as np
import json
import pickle
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns

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

# 训练逻辑回归模型
def train_logistic_regression(X_train, y_train, X_val=None, y_val=None):
    """训练逻辑回归模型"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"逻辑回归最佳参数: {grid_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

# 训练SVM模型
def train_svm(X_train, y_train, X_val=None, y_val=None):
    """训练SVM模型"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"SVM最佳参数: {grid_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

# 训练随机森林模型
def train_random_forest(X_train, y_train, X_val=None, y_val=None):
    """训练随机森林模型"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"随机森林最佳参数: {grid_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

# 训练梯度提升树模型
def train_gradient_boosting(X_train, y_train, X_val=None, y_val=None):
    """训练梯度提升树模型"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"梯度提升树最佳参数: {grid_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

# 训练朴素贝叶斯模型
def train_naive_bayes(X_train, y_train, X_val=None, y_val=None):
    """训练朴素贝叶斯模型，适用于文本分类任务"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 创建朴素贝叶斯模型
    nb_model = MultinomialNB()
    
    # 定义参数网格
    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(nb_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"朴素贝叶斯最佳参数: {grid_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

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
    
    # 计算精确率、召回率、F1值
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'confusion_matrix_{model.__class__.__name__}.png'))
    
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1值: {f1:.4f}")
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

# 训练AdaBoost模型
def train_adaboost(X_train, y_train, X_val=None, y_val=None):
    """训练AdaBoost模型，通过提升弱分类器的性能"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"AdaBoost最佳参数: {grid_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

# 训练Bagging模型
def train_bagging(X_train, y_train, X_val=None, y_val=None):
    """训练Bagging模型，通过多个基础模型的平均预测提高性能"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [10, 20, 50],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(BaggingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 在验证集上评估
    val_accuracy = best_model.score(X_val, y_val)
    y_pred = best_model.predict(X_val)
    
    print(f"Bagging最佳参数: {grid_search.best_params_}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return best_model

# 训练投票集成模型
def train_voting_ensemble(X_train, y_train, X_val=None, y_val=None):
    """训练投票集成模型，结合多个基础模型的预测"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 创建基础模型
    lr = LogisticRegression(C=10, solver='liblinear', max_iter=1000, random_state=42)
    svm = SVC(C=1, kernel='linear', probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    nb = MultinomialNB(alpha=0.1)
    
    # 创建集成模型
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr),
            ('svm', svm),
            ('rf', rf),
            ('gb', gb),
            ('nb', nb)
        ],
        voting='soft'  # 使用概率进行投票
    )
    
    # 训练模型
    start_time = time.time()
    ensemble.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 在验证集上评估
    val_accuracy = ensemble.score(X_val, y_val)
    y_pred = ensemble.predict(X_val)
    
    print(f"投票集成模型训练时间: {train_time:.2f}秒")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return ensemble

# 训练堆叠集成模型
def train_stacking_ensemble(X_train, y_train, X_val=None, y_val=None):
    """训练堆叠集成模型，使用元学习器组合多个基础模型的预测"""
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 创建基础模型
    estimators = [
        ('lr', LogisticRegression(C=10, solver='liblinear', max_iter=1000, random_state=42)),
        ('svm', SVC(C=1, kernel='linear', probability=True, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
        ('nb', MultinomialNB(alpha=0.1))
    ]
    
    # 创建堆叠集成模型，使用逻辑回归作为元学习器
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # 训练模型
    start_time = time.time()
    stacking.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 在验证集上评估
    val_accuracy = stacking.score(X_val, y_val)
    y_pred = stacking.predict(X_val)
    
    print(f"堆叠集成模型训练时间: {train_time:.2f}秒")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))
    
    return stacking

# 主函数
def main(feature_type='tfidf'):
    """主函数，训练和评估模型"""
    # 加载特征数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_features(feature_type)
    
    # 加载标签编码器
    with open(os.path.join(FEATURES_DIR, 'label_encoder.json'), 'r', encoding='utf-8') as f:
        label_encoder_data = json.load(f)
    
    # 创建标签编码器对象
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_encoder_data['classes'])
    
    # 训练基础模型
    print("\n训练逻辑回归模型...")
    lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    print("\n训练SVM模型...")
    svm_model = train_svm(X_train, y_train, X_val, y_val)
    
    print("\n训练随机森林模型...")
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    
    print("\n训练梯度提升树模型...")
    gb_model = train_gradient_boosting(X_train, y_train, X_val, y_val)
    
    print("\n训练朴素贝叶斯模型...")
    nb_model = train_naive_bayes(X_train, y_train, X_val, y_val)
    
    # 训练高级集成模型
    print("\n训练AdaBoost模型...")
    ada_model = train_adaboost(X_train, y_train, X_val, y_val)
    
    print("\n训练Bagging模型...")
    bag_model = train_bagging(X_train, y_train, X_val, y_val)
    
    print("\n训练投票集成模型...")
    voting_model = train_voting_ensemble(X_train, y_train, X_val, y_val)
    
    print("\n训练堆叠集成模型...")
    stacking_model = train_stacking_ensemble(X_train, y_train, X_val, y_val)
    
    # 在测试集上评估模型
    if X_test is not None and y_test is not None:
        # 评估基础模型
        print("\n在测试集上评估逻辑回归模型:")
        lr_accuracy, _, _ = evaluate_model(lr_model, X_test, y_test, label_encoder)
        
        print("\n在测试集上评估SVM模型:")
        svm_accuracy, _, _ = evaluate_model(svm_model, X_test, y_test, label_encoder)
        
        print("\n在测试集上评估随机森林模型:")
        rf_accuracy, _, _ = evaluate_model(rf_model, X_test, y_test, label_encoder)
        
        print("\n在测试集上评估梯度提升树模型:")
        gb_accuracy, _, _ = evaluate_model(gb_model, X_test, y_test, label_encoder)
        
        print("\n在测试集上评估朴素贝叶斯模型:")
        nb_accuracy, _, _ = evaluate_model(nb_model, X_test, y_test, label_encoder)
        
        # 评估高级集成模型
        print("\n在测试集上评估AdaBoost模型:")
        ada_accuracy, _, _ = evaluate_model(ada_model, X_test, y_test, label_encoder)
        
        print("\n在测试集上评估Bagging模型:")
        bag_accuracy, _, _ = evaluate_model(bag_model, X_test, y_test, label_encoder)
        
        print("\n在测试集上评估投票集成模型:")
        voting_accuracy, _, _ = evaluate_model(voting_model, X_test, y_test, label_encoder)
        
        print("\n在测试集上评估堆叠集成模型:")
        stacking_accuracy, _, _ = evaluate_model(stacking_model, X_test, y_test, label_encoder)
        
        # 比较所有模型的性能
        print("\n模型性能比较:")
        models = ['逻辑回归', 'SVM', '随机森林', '梯度提升树', '朴素贝叶斯', 'AdaBoost', 'Bagging', '投票集成', '堆叠集成']
        accuracies = [lr_accuracy, svm_accuracy, rf_accuracy, gb_accuracy, nb_accuracy, ada_accuracy, bag_accuracy, voting_accuracy, stacking_accuracy]
        
        for model_name, acc in zip(models, accuracies):
            print(f"{model_name}: {acc:.4f}")
        
        # 找出最佳模型
        best_idx = np.argmax(accuracies)
        print(f"\n最佳模型: {models[best_idx]}，准确率: {accuracies[best_idx]:.4f}")
    
    # 保存模型
    save_model(lr_model, 'logistic_regression')
    save_model(svm_model, 'svm')
    save_model(rf_model, 'random_forest')
    save_model(gb_model, 'gradient_boosting')
    save_model(nb_model, 'naive_bayes')
    save_model(ensemble_model, 'ensemble')

if __name__ == "__main__":
    main(feature_type='tfidf')