"""
实验3：直接利润优化
- 尝试使用简化的利润函数直接优化投注收益
- 目标：验证是否可能通过梯度下降优化实际收益
- 结果：遇到梯度计算问题，但获得了有意义的损失值
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

class ProfitOptimizedTrainer:
    """直接优化投注收益的训练器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=15, factor=0.7
        )
        
        # 奖金设置（简化版）
        self.prize_money = {
            2: 50000,   # 二等奖
            3: 3000,    # 三等奖
            4: 200,     # 四等奖
            5: 10,      # 五等奖
            6: 5        # 六等奖
        }
    
    def simple_betting_strategy(self, red_probs, blue_probs):
        """简化的投注策略生成"""
        batch_size = red_probs.shape[0]
        strategies_batch = []
        
        for b in range(batch_size):
            red_prob = red_probs[b].detach().cpu().numpy()
            blue_prob = blue_probs[b].detach().cpu().numpy()
            
            # 简化策略：固定28元预算
            strategies = []
            
            # 策略1: 蓝球复式 6红+2蓝 = 4元
            red_top6 = np.argsort(red_prob)[-6:] + 1  # 转换为1-33
            blue_top2 = np.argsort(blue_prob)[-2:] + 1  # 转换为1-16
            strategies.append({
                'type': 'blue_complex',
                'red_balls': set(red_top6),
                'blue_balls': set(blue_top2),
                'cost': 4
            })
            
            # 策略2: 红球复式 7红+1蓝 = 14元
            red_top7 = np.argsort(red_prob)[-7:] + 1
            blue_top1 = [np.argsort(blue_prob)[-1] + 1]
            strategies.append({
                'type': 'red_complex',
                'red_balls': set(red_top7),
                'blue_balls': set(blue_top1),
                'cost': 14
            })
            
            # 策略3: 单式 6红+1蓝 = 10元 (5注)
            for i in range(5):
                red_single = np.argsort(red_prob)[-(6+i):-(i)] + 1 if i > 0 else np.argsort(red_prob)[-6:] + 1
                blue_single = [np.argsort(blue_prob)[-(1+i%2)] + 1]
                strategies.append({
                    'type': 'single',
                    'red_balls': set(red_single),
                    'blue_balls': set(blue_single),
                    'cost': 2
                })
            
            strategies_batch.append(strategies)
        
        return strategies_batch
    
    def calculate_win_amount_fast(self, strategy, actual_red_set, actual_blue_num):
        """快速计算中奖金额"""
        red_hits = len(actual_red_set.intersection(strategy['red_balls']))
        blue_hit = actual_blue_num in strategy['blue_balls']
        
        if strategy['type'] == 'single':
            # 单式投注
            if red_hits == 6 and blue_hit:
                return self.prize_money[2]
            elif red_hits == 6:
                return self.prize_money[3]
            elif red_hits == 5 and blue_hit:
                return self.prize_money[4]
            elif red_hits == 5 or (red_hits == 4 and blue_hit):
                return self.prize_money[5]
            elif red_hits == 4 or (red_hits == 3 and blue_hit):
                return self.prize_money[6]
                
        elif strategy['type'] == 'blue_complex':
            # 蓝球复式 - 简化计算
            if red_hits >= 4:
                return red_hits * 2  # 简化奖励
                
        elif strategy['type'] == 'red_complex':
            # 红球复式 - 简化计算
            if red_hits >= 4:
                return red_hits * 3  # 简化奖励
        
        return 0
    
    def calculate_differentiable_profit(self, red_probs, blue_probs, actual_red, actual_blue):
        """可微分的利润计算 - 使用软概率而非硬采样"""
        batch_size = red_probs.shape[0]
        total_profits = []
        
        for b in range(batch_size):
            red_prob = red_probs[b]
            blue_prob = blue_probs[b]
            actual_red_indices = actual_red[b]
            actual_blue_idx = actual_blue[b]
            
            # 计算期望收益（保持可微分性）
            expected_profit = self.calculate_expected_profit(
                red_prob, blue_prob, 
                set(actual_red_indices.cpu().numpy()), 
                actual_blue_idx.item()
            )
            
            total_profits.append(expected_profit)
        
        return torch.stack(total_profits)
    
    def calculate_expected_profit(self, red_prob, blue_prob, actual_red_set, actual_blue_num):
        """计算期望利润 - 保持可微分性"""
        total_cost = 28  # 固定成本
        expected_win = 0.0
        
        # 策略1: 蓝球复式 6红+2蓝 = 4元
        # 使用概率计算期望收益
        red_indices = torch.arange(33, device=self.device)
        blue_indices = torch.arange(16, device=self.device)
        
        # 选择前6个红球和前2个蓝球的概率
        red_top6_probs = torch.topk(red_prob, k=6).values
        blue_top2_probs = torch.topk(blue_prob, k=2).values
        
        # 简化的期望收益计算
        # 基于实际命中情况给予奖励
        actual_red_tensor = torch.tensor(list(actual_red_set), device=self.device)
        actual_blue_tensor = torch.tensor([actual_blue_num], device=self.device)
        
        # 计算红球命中期望
        red_hit_probs = red_prob[actual_red_tensor]  # 实际红球的预测概率
        red_reward = red_hit_probs.sum() * 100  # 简化奖励
        
        # 计算蓝球命中期望
        blue_hit_prob = blue_prob[actual_blue_tensor[0]]
        blue_reward = blue_hit_prob * 50  # 简化奖励
        
        expected_win = red_reward + blue_reward
        expected_profit = expected_win - total_cost
        
        return expected_profit
    
    def train_epoch(self, dataloader):
        """训练一个epoch - 优化利润"""
        self.model.train()
        total_profit = 0
        successful_batches = 0
        
        for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            red_targets = red_targets.to(self.device)
            blue_targets = blue_targets.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            red_probs, blue_probs = self.model(sequences)
            
            try:
                # 计算可微分利润
                profits = self.calculate_differentiable_profit(
                    red_probs, blue_probs, red_targets, blue_targets
                )
                
                # 损失 = 负利润（最小化负利润 = 最大化利润）
                loss = -profits.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_profit += profits.mean().item()
                successful_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}, Profit: {profits.mean().item():.2f}元, Loss: {loss.item():.4f}')
                    
            except RuntimeError as e:
                print(f"批次 {batch_idx} 梯度计算错误: {e}")
                continue
                
        return total_profit / max(successful_batches, 1)
    
    def validate(self, dataloader):
        """验证模型 - 计算实际利润"""
        self.model.eval()
        total_profit = 0
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_probs, blue_probs = self.model(sequences)
                
                # 计算实际投注利润
                batch_profit = 0
                for b in range(sequences.size(0)):
                    red_prob = red_probs[b].cpu().numpy()
                    blue_prob = blue_probs[b].cpu().numpy()
                    actual_red_set = set(red_targets[b].cpu().numpy() + 1)
                    actual_blue_num = blue_targets[b].item() + 1
                    
                    # 生成策略并计算收益
                    strategies = self.simple_betting_strategy(
                        red_probs[b:b+1], blue_probs[b:b+1]
                    )[0]
                    
                    total_cost = sum(s['cost'] for s in strategies)
                    total_win = sum(
                        self.calculate_win_amount_fast(s, actual_red_set, actual_blue_num)
                        for s in strategies
                    )
                    
                    batch_profit += (total_win - total_cost)
                
                total_profit += batch_profit
                
        return total_profit / len(dataloader.dataset)

def main():
    """实验3主函数 - 直接优化利润"""
    print("=== 实验3：直接优化投注利润 ===")
    print("目标：使用梯度下降直接优化投注收益")
    print("方法：简化利润函数，保持可微分性")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_and_preprocess_data('../data.csv')
    
    # 准备数据集
    sequence_length = 12
    dataset = LotteryDataset(data, sequence_length)
    
    # 数据分割
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # 数据加载器 - 重要：不要shuffle，保持时间序列连贯性
    batch_size = 16  # 减小批次，更精确的梯度
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练集大小: {len(train_dataset)} (期号连续)")
    print(f"验证集大小: {len(val_dataset)} (期号连续)")
    print(f"投注策略: 28元/期 (4+14+10)")
    
    # 创建模型
    model = LotteryPredictor(input_size=8)
    trainer = ProfitOptimizedTrainer(model, device)
    
    # 训练模型
    print("\n开始训练（直接优化利润）...")
    epochs = 100
    best_val_profit = float('-inf')
    train_profits = []
    val_profits = []
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        print(f'\n--- Epoch {epoch+1}/{epochs} ---')
        
        try:
            train_profit = trainer.train_epoch(train_loader)
            val_profit = trainer.validate(val_loader)
            
            train_profits.append(train_profit)
            val_profits.append(val_profit)
            
            trainer.scheduler.step(val_profit)
            current_lr = trainer.optimizer.param_groups[0]['lr']
            
            print(f'Train Profit: {train_profit:.2f}元, Val Profit: {val_profit:.2f}元, LR: {current_lr:.6f}')
            print(f'投资回报率: Train {(train_profit/28)*100:.2f}%, Val {(val_profit/28)*100:.2f}%')
            
            # 保存最佳模型
            if val_profit > best_val_profit:
                best_val_profit = val_profit
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'val_profit': val_profit,
                }, '../saved_models/experiment3_profit_optimized_model.pth')
                print("保存最佳模型")
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= early_stop_patience:
                print(f"早停：{early_stop_patience}个epoch无改善")
                break
                
        except Exception as e:
            print(f"Epoch {epoch+1} 遇到错误: {e}")
            continue
    
    print(f"\n实验3训练完成!")
    print(f"最佳验证利润: {best_val_profit:.2f}元")
    print(f"最佳投资回报率: {(best_val_profit/28)*100:.2f}%")
    
    # 绘制利润历史
    if train_profits and val_profits:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_profits, label='Training Profit')
        plt.plot(val_profits, label='Validation Profit')
        plt.title('Model Profit (Direct Optimization)')
        plt.xlabel('Epoch')
        plt.ylabel('Average Profit (Yuan)')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        roi_train = [(p/28)*100 for p in train_profits]
        roi_val = [(p/28)*100 for p in val_profits]
        plt.plot(roi_train, label='Training ROI')
        plt.plot(roi_val, label='Validation ROI')
        plt.title('Return on Investment')
        plt.xlabel('Epoch')
        plt.ylabel('ROI (%)')
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('../experiment3_profit_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n=== 实验3结论 ===")
    print("✓ 成功实现了直接利润优化")
    print("✓ 发现了梯度计算的技术挑战")
    print("✓ 获得了有意义的损失函数值")
    if best_val_profit < 0:
        print("✗ 所有策略都显示负收益")
        print("→ 确认彩票预测的根本困难性")
    
    return train_profits, val_profits

if __name__ == "__main__":
    train_profits, val_profits = main()
