import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from models.model import SimpleClassifier
import pandas as pd

def evaluate_model(model_path, X_test, y_test):
    """
    评估模型性能
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和预处理器
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    scaler = checkpoint['scaler']
    
    # 初始化模型
    model = SimpleClassifier().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 预处理测试数据
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs > 0.5).float().cpu().numpy()
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return predictions

if __name__ == '__main__':
    # 这里需要提供测试数据
    # 在实际使用中，您可能需要从文件中加载测试数据
    # 例如：
    test_data = pd.read_csv('./data/sample_data.csv')
    X_test = test_data[['feature1', 'feature2']].values
    y_test = test_data['label'].values
    evaluate_model('model_and_scaler.pth', X_test, y_test)
    
    # print("请提供测试数据以进行评估")