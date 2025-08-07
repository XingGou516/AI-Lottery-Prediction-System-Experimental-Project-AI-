import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lottery_model import LotteryDataset, LotteryPredictor, load_and_preprocess_data, analyze_data

class ProfitOptimizedTrainer:
    """直接优化投注收益的训练器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=15, factor=0.7  # 最大化利润
        )
        
        # 奖金设置
        self.prize_money = {
            2: 50000,   # 二等奖：降低期望，更现实
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
            red_p = red_probs[b].detach().cpu().numpy()
            blue_p = blue_probs[b].detach().cpu().numpy()
            
            # 获取最高概率的号码
            red_top = np.argsort(red_p)[-10:][::-1] + 1  # 前10个红球
            blue_top = np.argsort(blue_p)[-3:][::-1] + 1  # 前3个蓝球
            
            strategies = []
            
            # 策略1: 蓝球复式 6红+2蓝 = 4元
            strategies.append({
                'type': 'blue_complex',
                'red_balls': set(red_top[:6]),
                'blue_balls': set(blue_top[:2]),
                'cost': 4
            })
            
            # 策略2: 5注单式 = 10元
            for i in range(5):
                strategies.append({
                    'type': 'single',
                    'red_balls': set(red_top[i:i+6]),
                    'blue_balls': set([blue_top[i % 2]]),
                    'cost': 2
                })
            
            # 策略3: 红球小复式 7红+1蓝 = 14元
            strategies.append({
                'type': 'red_complex',
                'red_balls': set(red_top[:7]),
                'blue_balls': set([blue_top[0]]),
                'cost': 14
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
                return self.prize_money[2]  # 二等奖
            elif red_hits == 5 and blue_hit:
                return self.prize_money[3]  # 三等奖
            elif red_hits == 5 or (red_hits == 4 and blue_hit):
                return self.prize_money[4]  # 四等奖
            elif red_hits == 4 or (red_hits == 3 and blue_hit):
                return self.prize_money[5]  # 五等奖
            elif blue_hit:
                return self.prize_money[6]  # 六等奖
            return 0
            
        elif strategy['type'] == 'blue_complex':
            # 蓝球复式 6红+2蓝
            if blue_hit:
                if red_hits == 6:
                    return self.prize_money[2]
                elif red_hits == 5:
                    return self.prize_money[3]
                elif red_hits == 4:
                    return self.prize_money[5]
                elif red_hits == 3:
                    return self.prize_money[5]
                else:
                    return self.prize_money[6]
            return 0
            
        elif strategy['type'] == 'red_complex':
            # 红球复式 7红+1蓝
            if red_hits >= 4:  # 至少4个红球才有意义
                combinations = 7 if red_hits == 6 else (6 if red_hits == 5 else 15)
                
                if red_hits == 6 and blue_hit:
                    return self.prize_money[2] * combinations
                elif red_hits == 6:
                    return self.prize_money[2] * combinations
                elif red_hits == 5 and blue_hit:
                    return self.prize_money[3] * 6
                elif red_hits == 5:
                    return self.prize_money[4] * 6
                elif red_hits == 4 and blue_hit:
                    return self.prize_money[5] * 15
                elif red_hits == 4:
                    return self.prize_money[5] * 15
            if blue_hit:
                return self.prize_money[6] * 21
            return 0
        
        return 0
    
    def calculate_batch_profit(self, red_probs, blue_probs, actual_red, actual_blue):
        """计算批次利润"""
        strategies_batch = self.simple_betting_strategy(red_probs, blue_probs)
        profits = []
        
        batch_size = len(strategies_batch)
        for b in range(batch_size):
            actual_red_set = set(actual_red[b].cpu().numpy())
            actual_blue_num = actual_blue[b].item()
            
            total_cost = 0
            total_win = 0
            
            for strategy in strategies_batch[b]:
                total_cost += strategy['cost']
                win = self.calculate_win_amount_fast(strategy, actual_red_set, actual_blue_num)
                total_win += win
            
            profit = total_win - total_cost
            profits.append(profit)
        
        return torch.tensor(profits, device=self.device, dtype=torch.float32)
    
    def train_epoch(self, dataloader):
        """训练一个epoch - 直接优化利润"""
        self.model.train()
        total_profit = 0
        batch_count = 0
        
        for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            red_targets = red_targets.to(self.device)
            blue_targets = blue_targets.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            red_logits, blue_logits = self.model(sequences)
            red_probs = torch.softmax(red_logits, dim=-1)
            blue_probs = torch.softmax(blue_logits, dim=-1)
            
            # 计算利润
            profits = self.calculate_batch_profit(red_probs, blue_probs, red_targets, blue_targets)
            
            # 损失 = 负平均利润
            loss = -profits.mean()
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_profit += profits.sum().item()
            batch_count += len(profits)
            
            if batch_idx % 30 == 0:
                avg_profit = profits.mean().item()
                print(f'  Batch {batch_idx:3d} | Avg Profit: {avg_profit:.2f}元 | Loss: {loss.item():.4f}')
        
        return total_profit / batch_count
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_profit = 0
        batch_count = 0
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_logits, blue_logits = self.model(sequences)
                red_probs = torch.softmax(red_logits, dim=-1)
                blue_probs = torch.softmax(blue_logits, dim=-1)
                
                profits = self.calculate_batch_profit(red_probs, blue_probs, red_targets, blue_targets)
                total_profit += profits.sum().item()
                batch_count += len(profits)
        
        return total_profit / batch_count

def main():
    """实验3主函数 - 直接优化利润"""
    print("=== 实验3：直接优化投注利润 ===")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_and_preprocess_data('data.csv')
    
    # 准备数据集
    sequence_length = 12
    dataset = LotteryDataset(data, sequence_length)
    
    # 数据分割
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # 数据加载器
    batch_size = 16  # 减小批次，更精确的梯度
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"投注策略: 28元/期 (4+10+14)")
    
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
        
        train_profit = trainer.train_epoch(train_loader)
        val_profit = trainer.validate(val_loader)
        
        train_profits.append(train_profit)
        val_profits.append(val_profit)
        
        trainer.scheduler.step(val_profit)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        print(f'Train Profit: {train_profit:.2f}元 | Val Profit: {val_profit:.2f}元 | LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_profit > best_val_profit:
            best_val_profit = val_profit
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_profit': train_profit,
                'val_profit': val_profit,
            }, 'experiment3_profit_optimized_model.pth')
            print(f'*** 最佳模型已保存! 验证利润: {val_profit:.2f}元 ***')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= early_stop_patience:
            print(f'\n早停触发! {early_stop_patience} 个epoch没有改善.')
            break
    
    print(f"\n实验3训练完成!")
    print(f"最佳验证利润: {best_val_profit:.2f}元")
    print(f"投资回报率: {(best_val_profit/28)*100:.2f}%")
    
    return train_profits, val_profits

if __name__ == "__main__":
    train_profits, val_profits = main()
