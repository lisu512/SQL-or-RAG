import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# 特征选择函数
def select_features(X_train, y_train, X_val=None, X_test=None, method='chi2', k=100):
    """使用不同的特征选择方法选择最重要的特征"""
    if method == 'chi2':
        # 使用卡方统计量选择特征
        selector = SelectKBest(chi2, k=k)
    elif method == 'mutual_info':
        # 使用互信息选择特征
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'tree_based':
        # 使用基于树的特征重要性选择特征
        selector = SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=42))
    else:
        raise ValueError(f"不支持的特征选择方法: {method}")
    
    # 拟合并转换训练集
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # 转换验证集和测试集（如果提供）
    X_val_selected = selector.transform(X_val) if X_val is not None else None
    X_test_selected = selector.transform(X_test) if X_test is not None else None
    
    print(f"特征选择完成，从{X_train.shape[1]}个特征中选择了{X_train_selected.shape[1]}个特征")
    
    return X_train_selected, X_val_selected, X_test_selected, selector

# 绘制学习曲线
def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """绘制学习曲线，分析模型的学习过程"""
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集得分")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="验证集得分")
    
    plt.xlabel("训练样本数")
    plt.ylabel("准确率")
    plt.title("学习曲线")
    plt.legend(loc="best")
    
    return plt

# 绘制验证曲线
def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5, scoring='accuracy'):
    """绘制验证曲线，分析模型参数的影响"""
    plt.figure(figsize=(10, 6))
    
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="训练集得分")
    plt.plot(param_range, test_scores_mean, 'o-', color="g", label="验证集得分")
    
    plt.xlabel(param_name)
    plt.ylabel("准确率")
    plt.title(f"验证曲线 ({param_name})")
    plt.legend(loc="best")
    
    return plt

# 详细评估模型性能
def evaluate_model_detailed(model, X_test, y_test, class_names=None):
    """详细评估模型性能，包括混淆矩阵、分类报告等"""
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 计算精确率、召回率、F1值
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # 生成混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    
    # 打印评估结果
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print("\n分类报告:")
    print(report)
    
    return accuracy, precision, recall, f1, cm, report

# 绘制ROC曲线
def plot_roc_curve(model, X_test, y_test, class_names=None):
    """绘制ROC曲线，评估模型的分类性能"""
    # 获取预测概率
    y_proba = model.predict_proba(X_test)
    
    # 将标签进行one-hot编码
    n_classes = len(class_names) if class_names is not None else y_proba.shape[1]
    y_test_bin = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        y_test_bin[i, y_test[i]] = 1
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制所有ROC曲线
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i] if class_names is not None else i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)')
    plt.ylabel('真正例率 (True Positive Rate)')
    plt.title('接收者操作特征曲线 (ROC)')
    plt.legend(loc="lower right")
    
    return plt, roc_auc