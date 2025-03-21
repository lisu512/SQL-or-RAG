import json
import pickle
import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
from src.data.preprocess import clean_text, tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

def load_model_and_vectorizer() -> Tuple[Any, TfidfVectorizer, list]:
    """加载模型和向量化器"""
    # 获取项目根目录路径
    root_dir = Path(__file__).resolve().parent.parent
    
    # 加载模型
    model_path = root_dir / 'models' / 'stacking_ensemble.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件未找到：{model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # 加载向量化器数据
    vectorizer_path = root_dir / 'data' / 'processed' / 'features' / 'tfidf_vectorizer.json'
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"向量化器文件未找到：{vectorizer_path}")
    
    with open(vectorizer_path, 'r', encoding='utf-8') as f:
        vectorizer_data = json.load(f)
    
    # 重建向量化器
    vectorizer = TfidfVectorizer(vocabulary=vectorizer_data['vocabulary'])
    if 'idf' in vectorizer_data and vectorizer_data['idf'] is not None:
        vectorizer.idf_ = np.array(vectorizer_data['idf'])
    
    # 加载标签编码器
    label_encoder_path = root_dir / 'data' / 'processed' / 'features' / 'label_encoder.json'
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"标签编码器文件未找到：{label_encoder_path}")
    
    with open(label_encoder_path, 'r', encoding='utf-8') as f:
        label_encoder_data = json.load(f)
    
    return model, vectorizer, label_encoder_data['classes']

def predict_intent(text: str, model=None, vectorizer=None, classes=None) -> Tuple[str, float, Dict[str, float]]:
    """预测文本的意图"""
    # 如果没有提供模型和向量化器，则加载它们
    if model is None or vectorizer is None or classes is None:
        model, vectorizer, classes = load_model_and_vectorizer()
    
    # 预处理文本
    processed_text = tokenize(clean_text(text))
    
    # 提取特征并转换为密集矩阵
    features = vectorizer.transform([processed_text]).toarray()
    
    # 预测
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]
    predicted_class = classes[prediction[0]]
    probability = max(probabilities)
    
    # 构建每个类别的概率字典
    class_probabilities = {cls: prob for cls, prob in zip(classes, probabilities)}
    
    return predicted_class, probability, class_probabilities

def display_results(text: str, intent: str, probability: float, class_probabilities: Dict[str, float]) -> None:
    """以美观的方式显示预测结果"""
    console = Console()
    
    # 创建结果表格
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("类别", style="cyan")
    table.add_column("概率", justify="right", style="green")
    
    # 添加每个类别的概率
    for cls, prob in sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True):
        prob_str = f"{prob:.4f}"
        if cls == intent:
            # 高亮显示预测的类别
            table.add_row(f"[bold yellow]{cls}[/bold yellow]", f"[bold yellow]{prob_str}[/bold yellow]")
        else:
            table.add_row(cls, prob_str)
    
    # 显示输入文本和预测结果
    console.print("\n[bold blue]输入文本:[/bold blue]", text)
    console.print("\n[bold blue]预测结果:[/bold blue]")
    console.print(table)

def main() -> None:
    console = Console()
    console.print("[bold green]欢迎使用SQL/RAG意图分类模型测试程序！[/bold green]")
    console.print("[italic]请输入您想测试的文本（输入'q'退出）：[/italic]")
    
    try:
        # 预加载模型和向量化器
        with Progress() as progress:
            task = progress.add_task("[cyan]正在加载模型...", total=1)
            model, vectorizer, classes = load_model_and_vectorizer()
            # 预加载jieba词典
            _ = tokenize("预加载")
            progress.update(task, completed=1)
        
        while True:
            try:
                text = input("\n请输入文本: ").strip()
                if not text:
                    console.print("[yellow]请输入有效的文本！[/yellow]")
                    continue
                if text.lower() == 'q':
                    break
                
                with Progress() as progress:
                    task = progress.add_task("[cyan]正在预测...", total=1)
                    intent, probability, class_probabilities = predict_intent(text, model, vectorizer, classes)
                    progress.update(task, completed=1)
                
                display_results(text, intent, probability, class_probabilities)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]程序已中断[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]预测过程中出现错误：{str(e)}[/red]")
    
    except FileNotFoundError as e:
        console.print(f"\n[red]错误：{str(e)}[/red]")
        console.print("[yellow]请确保模型文件和特征文件已正确放置在指定目录中。[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]程序初始化失败：{str(e)}[/red]")
        sys.exit(1)
    
    console.print("\n[bold green]感谢使用！再见！[/bold green]")

if __name__ == "__main__":
    main()