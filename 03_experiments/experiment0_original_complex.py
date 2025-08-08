"""
实验0：原始复杂损失函数
- 尝试直接优化投注收益
- 使用复杂的利润计算作为损失函数
- 结果：梯度计算困难，训练不稳定
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_models'))
from lottery_model import LotteryDataset, LotteryPredictor, load_and_preprocess_data, analyze_data

class OriginalLotteryTrainer:
    """原始训练器 - 复杂损失函数"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=10, factor=0.5
        )
        
        # 奖金设置
        self.prize_money = {
            2: 100000,  # 二等奖
            3: 3000,    # 三等奖
            4: 200,     # 四等奖
            5: 10,      # 五等奖
            6: 5        # 六等奖
        }
    
    def calculate_profit_loss(self, red_probs, blue_probs, actual_red, actual_blue):
        """计算基于投注策略的实际盈亏 - 原始复杂版本"""
        batch_size = red_probs.shape[0]
        total_profits = []
        
        for b in range(batch_size):
            # 注意：这里使用detach()会切断梯度链，导致梯度计算问题
            red_prob = red_probs[b].detach().cpu().numpy()
            blue_prob = blue_probs[b].detach().cpu().numpy()
            actual_red_set = set(actual_red[b].cpu().numpy() + 1)
            actual_blue_num = actual_blue[b].item() + 1
            
            # 生成投注策略
            strategies = self.generate_betting_strategy_tensor(red_prob, blue_prob, budget=30)
            
            # 计算总收益
            total_cost = sum(s['cost'] for s in strategies)
            total_win = 0
            
            for strategy in strategies:
                win_amount = self.calculate_win_amount(strategy, actual_red_set, actual_blue_num)
                total_win += win_amount
            
            profit = total_win - total_cost
            total_profits.append(profit)
        
        # 这里创建的tensor没有梯度信息，会导致"does not require grad"错误
        return torch.tensor(total_profits, device=self.device, dtype=torch.float32, requires_grad=False)
    
    def generate_betting_strategy_tensor(self, red_probs, blue_probs, budget=30):
        """生成投注策略（张量版本）"""
        red_indices = np.argsort(red_probs)[::-1]
        blue_indices = np.argsort(blue_probs)[::-1]
        red_numbers = red_indices + 1
        blue_numbers = blue_indices + 1
        
        strategies = []
        
        # 策略1: 蓝球复式 6红+3蓝 = 6元
        if budget >= 6:
            strategies.append({
                'type': 'blue_complex',
                'red_balls': set(red_numbers[:6]),
                'blue_balls': set(blue_numbers[:3]),
                'cost': 6
            })
            budget -= 6
        
        # 策略2: 红球复式 7红+1蓝 = 14元
        if budget >= 14:
            strategies.append({
                'type': 'red_complex',
                'red_balls': set(red_numbers[:7]),
                'blue_balls': set(blue_numbers[:1]),
                'cost': 14
            })
            budget -= 14
        
        # 策略3: 单式投注
        single_bets = budget // 2
        for i in range(min(single_bets, 5)):
            strategies.append({
                'type': 'single',
                'red_balls': set(red_numbers[i:i+6]),
                'blue_balls': set([blue_numbers[i % len(blue_numbers)]]),
                'cost': 2
            })
        
        return strategies
    
    def calculate_win_amount(self, strategy, actual_red_set, actual_blue_num):
        """计算中奖金额"""
        red_hits = len(actual_red_set.intersection(strategy['red_balls']))
        blue_hit = actual_blue_num in strategy['blue_balls']
        
        if strategy['type'] == 'single':
            # 单式投注
            if red_hits == 6 and blue_hit:
                return self.prize_money[2]  # 二等奖（一等奖需要特殊处理）
            elif red_hits == 6:
                return self.prize_money[3]  # 三等奖
            elif red_hits == 5 and blue_hit:
                return self.prize_money[4]  # 四等奖
            elif red_hits == 5 or (red_hits == 4 and blue_hit):
                return self.prize_money[5]  # 五等奖
            elif red_hits == 4 or (red_hits == 3 and blue_hit):
                return self.prize_money[6]  # 六等奖
                
        elif strategy['type'] == 'blue_complex':
            # 蓝球复式投注
            if red_hits == 6 and blue_hit:
                return self.prize_money[2]
            elif red_hits == 6:
                return self.prize_money[3] * len(strategy['blue_balls'])
            elif red_hits == 5 and blue_hit:
                return self.prize_money[4] * len(strategy['blue_balls'])
            elif red_hits == 5:
                return self.prize_money[5] * len(strategy['blue_balls'])
            elif red_hits == 4 and blue_hit:
                return self.prize_money[5] * len(strategy['blue_balls'])
            elif red_hits == 4:
                return self.prize_money[6] * len(strategy['blue_balls'])
                
        elif strategy['type'] == 'red_complex':
            # 红球复式投注
            if red_hits == 6 and blue_hit:
                return self.prize_money[2]
            elif red_hits == 6:
                return self.prize_money[3]
            elif red_hits == 5 and blue_hit:
                return self.prize_money[4]
            elif red_hits == 5:
                return self.prize_money[5]
            elif red_hits == 4 and blue_hit:
                return self.prize_money[5]
            elif red_hits == 4:
                return self.prize_money[6]
        
        return 0
    
    def train_epoch(self, dataloader):
        """训练一个epoch - 使用复杂损失函数"""
        self.model.train()
        total_profit = 0
        failed_batches = 0
        
        for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            red_targets = red_targets.to(self.device)
            blue_targets = blue_targets.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            red_probs, blue_probs = self.model(sequences)
            
            try:
                # 尝试计算利润损失（目标是最大化利润）
                profits = self.calculate_profit_loss(red_probs, blue_probs, red_targets, blue_targets)
                
                # 损失 = 负利润（最小化负利润 = 最大化利润）
                loss = -profits.mean()
                
                # 检查loss是否有梯度
                if not loss.requires_grad:
                    raise RuntimeError("Loss tensor does not require gradients")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_profit += profits.mean().item()
                
            except RuntimeError as e:
                failed_batches += 1
                print(f"梯度计算错误: {e}")
                
                # Fallback: 使用简单的交叉熵损失
                red_loss = self.criterion(red_probs.view(-1, 33), red_targets.view(-1))
                blue_loss = self.criterion(blue_probs, blue_targets)
                fallback_loss = red_loss + blue_loss
                
                fallback_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 估算利润（用于统计）
                estimated_profit = -fallback_loss.item() * 10  # 粗略估算
                total_profit += estimated_profit
                
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}, 使用fallback损失，估算利润: {estimated_profit:.2f}元')
                continue
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Profit: {profits.mean().item():.2f}元')
                
        print(f"本epoch失败batch数: {failed_batches}/{len(dataloader)}")
        return total_profit / len(dataloader)
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_profit = 0
        failed_batches = 0
        
        with torch.no_grad():
            for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_probs, blue_probs = self.model(sequences)
                
                try:
                    profits = self.calculate_profit_loss(red_probs, blue_probs, red_targets, blue_targets)
                    total_profit += profits.mean().item()
                except Exception as e:
                    failed_batches += 1
                    # 使用fallback估算
                    red_loss = self.criterion(red_probs.view(-1, 33), red_targets.view(-1))
                    blue_loss = self.criterion(blue_probs, blue_targets)
                    estimated_profit = -(red_loss + blue_loss).item() * 10
                    total_profit += estimated_profit
                    continue
                
        if failed_batches > 0:
            print(f"验证中失败batch数: {failed_batches}/{len(dataloader)}")
        return total_profit / len(dataloader)

def main():
    """实验0主函数 - 原始复杂损失函数"""
    print("=== 实验0：原始复杂损失函数 ===")
    print("目标：直接优化投注收益")
    print("预期问题：梯度计算困难，训练不稳定")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_and_preprocess_data('../data.csv')
    analyze_data(data)
    
    # 准备数据集
    sequence_length = 12
    dataset = LotteryDataset(data, sequence_length)
    
    # 数据分割
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # 数据加载器 - 重要：不要shuffle，保持时间序列连贯性
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = LotteryPredictor(input_size=8)
    trainer = OriginalLotteryTrainer(model, device)
    
    # 尝试训练（预期会遇到问题）
    print("\n开始训练（预期会遇到梯度问题）...")
    epochs = 5  # 减少epoch数，快速演示问题
    best_val_profit = float('-inf')
    train_profits = []
    val_profits = []
    total_failed_batches = 0
    
    for epoch in range(epochs):
        print(f'\n--- Epoch {epoch+1}/{epochs} ---')
        
        try:
            train_profit = trainer.train_epoch(train_loader)
            val_profit = trainer.validate(val_loader)
            
            train_profits.append(train_profit)
            val_profits.append(val_profit)
            
            print(f'Train Profit: {train_profit:.2f}元')
            print(f'Val Profit: {val_profit:.2f}元')
            
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                # 创建保存目录
                os.makedirs('../saved_models', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_profit': val_profit,
                }, '../saved_models/experiment0_original_model.pth')
                
        except Exception as e:
            print(f"训练过程中遇到严重错误: {e}")
            print("这证实了复杂损失函数的根本问题")
            break
        
        # 检查是否所有batch都失败了
        if hasattr(trainer, 'last_failed_batches'):
            total_failed_batches += trainer.last_failed_batches
    
    print(f"\n=== 实验0结论 ===")
    print(f"问题分析:")
    print(f"1. 梯度计算困难：复杂的利润计算包含太多非可微分操作")
    print(f"2. detach()操作：切断了梯度链，导致'does not require grad'错误")
    print(f"3. 数值不稳定：复杂的策略计算导致梯度消失/爆炸")
    print(f"4. 训练效率低：大量batch失败，需要fallback机制")
    print(f"")
    print(f"技术原因:")
    print(f"- 使用了太多numpy操作和Python循环")
    print(f"- 投注策略计算无法向量化")
    print(f"- 利润计算包含大量条件判断")
    print(f"")
    print(f"解决方案:")
    print(f"- 实验1：使用简单的交叉熵损失")
    print(f"- 实验1.5：多标签分类方法")
    print(f"- 实验2：约束优化方法")
    print(f"")
    print(f"性能统计:")
    print(f"- 最佳验证利润: {best_val_profit:.2f}元")
    print(f"- 训练完成度: {len(train_profits)}/{epochs} epochs")
    if train_profits:
        print(f"- 平均训练利润: {np.mean(train_profits):.2f}元")
    if val_profits:
        print(f"- 平均验证利润: {np.mean(val_profits):.2f}元")
    
    return train_profits, val_profits

if __name__ == "__main__":
    train_profits, val_profits = main()
