import os
import numpy as np
import json
import pickle
import time
import torch
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import subprocess
import sys

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 检查RAPIDS cuML是否已安装，如果没有则尝试安装
def check_install_rapids():
    try:
        import cuml
        print("RAPIDS cuML已安装，版本:", cuml.__version__)
        return True
    except ImportError:
        print("RAPIDS cuML未安装，尝试安装...")
        try:
            # 检查CUDA版本
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                print(f"检测到CUDA版本: {cuda_version}")
                # 安装与CUDA版本兼容的RAPIDS
                if cuda_version.startswith('11.'):
                    rapids_version = '23.08'
                elif cuda_version.startswith('12.'):
                    rapids_version = '23.10'
                else:
                    print(f"不支持的CUDA版本: {cuda_version}，将使用CPU训练")
                    return False
                
                print(f"正在安装RAPIDS {rapids_version}...")
                # 使用pip安装RAPIDS
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                                          f'cuml-cu{cuda_version[:2]}=={rapids_version}', 
                                          '--extra-index-url=https://pypi.nvidia.com'])
                    import cuml
                    print("RAPIDS cuML安装成功，版本:", cuml.__version__)
                    return True
                except Exception as e:
                    print(f"安装RAPIDS失败: {e}")
                    print("将使用CPU训练")
                    return False
            else:
                print("未检测到CUDA，将使用CPU训练")
                return False
        except Exception as e:
            print(f"安装RAPIDS时出错: {e}")
            print("将使用CPU训练")
            return False

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 检查并安装RAPIDS
use_rapids = check_install_rapids() if torch.cuda.is_available() else False

# 将NumPy数组转换为PyTorch张量并移动到GPU
def to_device(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    return data

# 定义路径
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed')
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 加载特征数据
def load_features(feature_type='tfidf'):
    """加载特征数据并转移到GPU"""
    X_train = to_device(np.load(os.path.join(FEATURES_DIR, f'X_train_{feature_type}.npy')))
    y_train = to_device(np.load(os.path.join(FEATURES_DIR, 'y_train.npy')))
    
    # 检查是否有验证集和测试集
    X_val, y_val = None, None
    X_test, y_test = None, None
    
    if os.path.exists(os.path.join(FEATURES_DIR, f'X_val_{feature_type}.npy')):
        X_val = to_device(np.load(os.path.join(FEATURES_DIR, f'X_val_{feature_type}.npy')))
        y_val = to_device(np.load(os.path.join(FEATURES_DIR, 'y_val.npy')))
    
    if os.path.exists(os.path.join(FEATURES_DIR, f'X_test_{feature_type}.npy')):
        X_test = to_device(np.load(os.path.join(FEATURES_DIR, f'X_test_{feature_type}.npy')))
        y_test = to_device(np.load(os.path.join(FEATURES_DIR, 'y_test.npy')))
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# 评估模型在测试集上的性能
def evaluate_model(model, X_test, y_test, label_encoder, batch_size=1024):
    """评估模型在测试集上的性能，支持批处理评估"""
    # 如果数据是PyTorch张量，转换回NumPy数组
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
        
    # 使用批处理进行预测以优化内存使用
    predictions = []
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        X_batch = X_test[i:batch_end]
        batch_pred = model.predict(X_batch)
        predictions.extend(batch_pred)
    
    # 转换预测结果为数组
    y_pred = np.array(predictions)
    
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
def train_stacking_ensemble(X_train, y_train, X_val=None, y_val=None, batch_size=1024):
    """训练堆叠集成模型，使用元学习器组合多个基础模型的预测"""
    # 如果数据是PyTorch张量，转换回NumPy数组
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()
    if X_val is not None and isinstance(X_val, torch.Tensor):
        X_val = X_val.cpu().numpy()
    if y_val is not None and isinstance(y_val, torch.Tensor):
        y_val = y_val.cpu().numpy()
        
    # 如果没有验证集，从训练集分割一部分作为验证集
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 创建并优化基础模型，使用GPU加速配置
    if use_rapids:
        from cuml.linear_model import LogisticRegression
        from cuml.svm import SVC
        from cuml.ensemble import RandomForestClassifier
        
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, tol=1e-4, C=1.0)),
            ('svm', SVC(probability=True, kernel='rbf', C=10.0, gamma='scale', tol=1e-4)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2))
        ]
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        
        estimators = [
            ('lr', LogisticRegression(max_iter=1000, tol=1e-4, C=1.0, random_state=42)),  # 恢复迭代次数
            ('svm', SVC(probability=False, kernel='rbf', C=1.0, gamma='scale', random_state=42)),  # 改回RBF核
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42))  # 增加树的数量
        ]
    
    # 创建堆叠集成模型，使用更小的元学习器
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 增加交叉验证折数
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        cv=cv,
        n_jobs=-1  # 启用并行处理
    )
    
    # 扩展参数搜索空间（增加类别权重相关参数）
    param_distributions = {
        'lr__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'lr__class_weight': [None, 'balanced'],
        'svm__C': [0.1, 1.0, 10.0, 100.0, 500.0],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'rf__n_estimators': [150, 200, 250],  # 增加树的数量
        'rf__max_depth': [15, 20, 25],  # 增加深度
        'rf__class_weight': [None, 'balanced'],
        'final_estimator__n_estimators': [150, 200, 250],
        'final_estimator__max_depth': [15, 20, 25],
        'final_estimator__class_weight': [None, 'balanced']
    }
    
    # 增强随机搜索配置
    random_search = RandomizedSearchCV(
        stacking,
        param_distributions,
        n_iter=40,  # 翻倍搜索次数
        cv=8,  # 增加交叉验证折数
        scoring=['accuracy', 'f1_weighted', 'recall', 'precision'],
        refit='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    # 训练模型，使用批处理以优化内存使用
    start_time = time.time()
    
    # 确保数据在GPU上（如果可用）
    if torch.cuda.is_available():
        print("\n使用GPU进行训练...")
        # 设置CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 显示GPU信息
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"训练前GPU内存使用情况:")
        print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # 启用混合精度训练
        from torch.cuda.amp import autocast
        with autocast():
            random_search.fit(X_train, y_train)
    else:
        random_search.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    print(f"\n训练完成！总耗时: {train_time:.2f}秒")
    print("最佳参数组合:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"最佳交叉验证得分: {random_search.best_score_:.4f}")
    
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
def main(feature_type='tfidf', batch_size=1024):
    """主函数，训练和评估堆叠集成模型"""
    # 设置GPU内存管理
    if torch.cuda.is_available():
        # 清空GPU缓存
        torch.cuda.empty_cache()
        # 启用cuDNN自动调优
        torch.backends.cudnn.benchmark = True
        # 显示GPU信息
        print(f"\nGPU信息:")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前GPU内存使用情况:")
        print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # 根据GPU内存大小调整批处理大小
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory_gb < 4:  # 小于4GB的GPU
            batch_size = min(batch_size, 256)
        elif total_memory_gb < 8:  # 4-8GB的GPU
            batch_size = min(batch_size, 512)
        print(f"根据GPU内存大小({total_memory_gb:.1f}GB)，设置批处理大小为: {batch_size}")
    else:
        print("\n警告: 未检测到GPU，将使用CPU进行训练，这可能会很慢。")
    
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
    stacking_model = train_stacking_ensemble(X_train, y_train, X_val, y_val, batch_size=batch_size)
    
    # 在测试集上评估模型
    if X_test is not None and y_test is not None:
        print("\n在测试集上评估堆叠集成模型:")
        evaluate_model(stacking_model, X_test, y_test, label_encoder, batch_size=batch_size)
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n训练后GPU内存使用情况:")
            print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 保存模型
    save_model(stacking_model, 'stacking_ensemble')

if __name__ == "__main__":
    main(feature_type='tfidf')