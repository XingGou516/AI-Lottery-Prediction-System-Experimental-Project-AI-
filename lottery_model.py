import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class LotteryDataset(Dataset):
    """彩票数据集类"""
    def __init__(self, data, sequence_length=12):
        self.data = data
        self.sequence_length = sequence_length
        self.samples = []
        self.targets_red = []
        self.targets_blue = []
        
        self._prepare_data()
    
    def _prepare_data(self):
        """准备训练数据"""
        for i in range(len(self.data) - self.sequence_length):
            # 输入序列：过去sequence_length期的数据 (包含期号)
            sequence = self.data[i:i+self.sequence_length]
            
            # 目标：下一期的红球和蓝球
            target_period = self.data[i+self.sequence_length]
            
            # 提取红球和蓝球 (data包含期号，共8列)
            red_balls = target_period[1:7]  # 第2-7列为红球 (索引1-6)
            blue_ball = target_period[7]    # 第8列为蓝球 (索引7)
            
            self.samples.append(sequence)
            self.targets_red.append(red_balls)
            self.targets_blue.append(blue_ball)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.samples[idx])
        red_target = torch.LongTensor(self.targets_red[idx]) - 1  # 转换为0-32范围
        blue_target = torch.LongTensor([self.targets_blue[idx]]) - 1  # 转换为0-15范围
        
        return sequence, red_target, blue_target

class LotteryPredictor(nn.Module):
    """彩票预测网络"""
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.2):  # 改回8
        super(LotteryPredictor, self).__init__()
        
        # 共享的LSTM特征提取层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 红球预测分支
        self.red_branch = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 33)  # 33个红球号码
        )
        
        # 蓝球预测分支  
        self.blue_branch = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # 16个蓝球号码
        )
        
    def forward(self, x):
        # LSTM特征提取
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 红球和蓝球预测
        red_probs = torch.softmax(self.red_branch(last_output), dim=-1)
        blue_probs = torch.softmax(self.blue_branch(last_output), dim=-1)
        
        return red_probs, blue_probs

def load_and_preprocess_data(csv_file):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(csv_file, header=None)
    
    print(f"数据形状: {df.shape}")
    print("数据前5行:")
    print(df.head())
    
    # 确保所有列都是数值型
    for col in range(8):  # 第1-8列都处理
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 删除包含NaN的行
    df = df.dropna()
    
    print(f"清理后数据形状: {df.shape}")
    
    # 对期号进行归一化处理 (保留所有列，包括期号)
    data = df.values
    
    # 期号归一化到0-1范围
    period_min = data[:, 0].min()
    period_max = data[:, 0].max()
    data[:, 0] = (data[:, 0] - period_min) / (period_max - period_min)
    
    print(f"期号范围: {period_min} - {period_max} (已归一化)")
    
    return data

def analyze_data(data):
    """分析数据分布"""
    print("\n=== 数据分析 ===")
    
    # 期号分析
    periods = data[:, 0]
    print(f"期号范围 (归一化后): {periods.min():.3f} - {periods.max():.3f}")
    
    # 红球分析
    red_balls = data[:, 1:7].flatten()  # 第2-7列为红球
    print(f"红球范围: {red_balls.min()} - {red_balls.max()}")
    print(f"红球平均值: {red_balls.mean():.2f}")
    
    # 蓝球分析
    blue_balls = data[:, 7]  # 第8列为蓝球
    print(f"蓝球范围: {blue_balls.min()} - {blue_balls.max()}")
    print(f"蓝球平均值: {blue_balls.mean():.2f}")
    
    # 号码频率统计
    print("\n红球出现频率前10:")
    red_freq = pd.Series(red_balls).value_counts().head(10)
    print(red_freq)
    
    print("\n蓝球出现频率:")
    blue_freq = pd.Series(blue_balls).value_counts().sort_index()
    print(blue_freq)
