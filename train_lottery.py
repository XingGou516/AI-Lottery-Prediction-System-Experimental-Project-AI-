import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lottery_model import LotteryDataset, LotteryPredictor, load_and_preprocess_data, analyze_data

class LotteryTrainer:
    """彩票预测训练器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=10, factor=0.5  # 改为max，优化收益
        )
        
        # 奖金设置
        self.prize_money = {
            2: 100000,  # 二等奖：6红+0蓝
            3: 3000,    # 三等奖：5红+1蓝
            4: 200,     # 四等奖：5红+0蓝 或 4红+1蓝
            5: 10,      # 五等奖：4红+0蓝 或 3红+1蓝
            6: 5        # 六等奖：0-6红+1蓝
        }
    
    def calculate_profit_loss(self, red_probs, blue_probs, actual_red, actual_blue):
        """计算基于投注策略的实际盈亏"""
        batch_size = red_probs.shape[0]
        total_profits = []
        
        for b in range(batch_size):
            # 生成投注策略 (需要detach才能转numpy)
            strategies = self.generate_betting_strategy_tensor(
                red_probs[b].detach().cpu().numpy(), 
                blue_probs[b].detach().cpu().numpy()
            )
            
            # 计算实际中奖情况
            actual_red_set = set(actual_red[b].cpu().numpy())
            actual_blue_num = actual_blue[b].item()
            
            total_cost = 0
            total_win = 0
            
            for strategy in strategies:
                total_cost += strategy['cost']
                win_amount = self.calculate_win_amount(
                    strategy, actual_red_set, actual_blue_num
                )
                total_win += win_amount
            
            profit = total_win - total_cost
            total_profits.append(profit)
        
        return torch.tensor(total_profits, device=self.device, dtype=torch.float32)
    
    def generate_betting_strategy_tensor(self, red_probs, blue_probs, budget=30):
        """生成投注策略（张量版本）"""
        red_indices = np.argsort(red_probs)[::-1]
        blue_indices = np.argsort(blue_probs)[::-1]
        red_numbers = red_indices + 1
        blue_numbers = blue_indices + 1
        
        strategies = []
        
        # 策略1: 蓝球复式 6红+3蓝 = 6元
        strategies.append({
            'type': 'blue_complex',
            'red_balls': set(red_numbers[:6]),
            'blue_balls': set(blue_numbers[:3]),
            'cost': 6
        })
        
        # 策略2: 红球复式 7红+1蓝 = 14元
        strategies.append({
            'type': 'red_complex',
            'red_balls': set(red_numbers[:7]),
            'blue_balls': set([blue_numbers[0]]),
            'cost': 14
        })
        
        # 策略3: 5注单式投注 = 10元
        for i in range(5):
            strategies.append({
                'type': 'single',
                'red_balls': set(red_numbers[i:i+6]),
                'blue_balls': set([blue_numbers[i % 3]]),
                'cost': 2
            })
        
        return strategies
    
    def calculate_win_amount(self, strategy, actual_red_set, actual_blue_num):
        """计算单个策略的中奖金额"""
        if strategy['type'] == 'blue_complex':
            # 蓝球复式：6红+3蓝
            if actual_blue_num in strategy['blue_balls']:
                red_hits = len(actual_red_set.intersection(strategy['red_balls']))
                if red_hits >= 3:  # 至少3红+1蓝才有奖
                    if red_hits == 6:
                        return self.prize_money[2]  # 二等奖
                    elif red_hits == 5:
                        return self.prize_money[3]  # 三等奖
                    elif red_hits == 4:
                        return self.prize_money[5]  # 五等奖
                    elif red_hits == 3:
                        return self.prize_money[5]  # 五等奖
                else:
                    return self.prize_money[6]  # 六等奖（只中蓝球）
            return 0
            
        elif strategy['type'] == 'red_complex':
            # 红球复式：7红+1蓝
            red_hits = len(actual_red_set.intersection(strategy['red_balls']))
            blue_hit = actual_blue_num in strategy['blue_balls']
            
            if red_hits == 6 and blue_hit:
                return self.prize_money[2] * 7  # 二等奖×7注
            elif red_hits == 6:
                return self.prize_money[2] * 7  # 二等奖×7注
            elif red_hits == 5 and blue_hit:
                return self.prize_money[3] * 6  # 三等奖×6注
            elif red_hits == 5:
                return self.prize_money[4] * 6  # 四等奖×6注
            elif red_hits == 4 and blue_hit:
                return self.prize_money[5] * 15  # 五等奖×15注
            elif red_hits == 4:
                return self.prize_money[5] * 15  # 五等奖×15注
            elif blue_hit:
                return self.prize_money[6] * 21  # 六等奖×21注
            return 0
            
        else:  # single
            # 单式：6红+1蓝
            red_hits = len(actual_red_set.intersection(strategy['red_balls']))
            blue_hit = actual_blue_num in strategy['blue_balls']
            
            if red_hits == 6 and blue_hit:
                return self.prize_money[2]  # 二等奖
            elif red_hits == 5 and blue_hit:
                return self.prize_money[3]  # 三等奖
            elif red_hits == 5:
                return self.prize_money[4]  # 四等奖
            elif red_hits == 4 and blue_hit:
                return self.prize_money[5]  # 五等奖
            elif red_hits == 4:
                return self.prize_money[5]  # 五等奖
            elif red_hits == 3 and blue_hit:
                return self.prize_money[5]  # 五等奖
            elif blue_hit:
                return self.prize_money[6]  # 六等奖
            return 0
        
    def train_epoch(self, dataloader):
        """训练一个epoch - 基于投注盈亏的损失函数"""
        self.model.train()
        total_profit = 0
        
        for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            red_targets = red_targets.to(self.device)
            blue_targets = blue_targets.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            red_probs, blue_probs = self.model(sequences)
            
            # 计算基于投注策略的实际盈亏
            profits = self.calculate_profit_loss(red_probs, blue_probs, red_targets, blue_targets)
            
            # 损失函数：负的平均利润（因为我们要最大化利润）
            loss = -profits.mean()
            
            # 添加少量交叉熵正则化，保持预测合理性
            red_ce_loss = 0
            for i in range(6):
                red_ce_loss += self.criterion(red_probs, red_targets[:, i])
            red_ce_loss /= 6
            blue_ce_loss = self.criterion(blue_probs, blue_targets)
            
            # 总损失：主要优化利润，辅助交叉熵正则化
            total_loss = loss + 0.1 * (red_ce_loss + blue_ce_loss)
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_profit += profits.mean().item()
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx:3d} | Avg Profit: {profits.mean().item():.2f}元 | Loss: {total_loss.item():.4f}')
                
        return total_profit / len(dataloader)  # 返回平均利润
    
    def validate(self, dataloader):
        """验证模型 - 计算平均利润"""
        self.model.eval()
        total_profit = 0
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_probs, blue_probs = self.model(sequences)
                
                # 计算利润
                profits = self.calculate_profit_loss(red_probs, blue_probs, red_targets, blue_targets)
                total_profit += profits.mean().item()
                
        return total_profit / len(dataloader)

def plot_training_history(train_profits, val_profits):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_profits, label='Training Profit')
    plt.plot(val_profits, label='Validation Profit')
    plt.title('Model Profit')
    plt.xlabel('Epoch')
    plt.ylabel('Average Profit (元)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_profits[-50:], label='Training Profit (Last 50 epochs)')
    plt.plot(val_profits[-50:], label='Validation Profit (Last 50 epochs)')
    plt.title('Model Profit (Recent)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Profit (元)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主训练函数"""
    print("=== 彩票预测模型训练 ===")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_and_preprocess_data('data.csv')
    
    # 数据分析
    analyze_data(data)
    
    # 准备数据集
    sequence_length = 12
    dataset = LotteryDataset(data, sequence_length)
    
    # 数据分割：前80%训练，后20%验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # 按时间顺序分割（而不是随机分割）
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    # 数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"序列长度: {sequence_length}")
    print(f"批次大小: {batch_size}")
    
    # 创建模型
    model = LotteryPredictor(input_size=8)  # 期号+6个红球+1个蓝球 = 8维
    trainer = LotteryTrainer(model, device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模型
    print("\n开始训练...")
    epochs = 100
    best_val_profit = float('-inf')  # 改为最大化利润
    train_profits = []  # 改为记录利润
    val_profits = []
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        print(f'\n--- Epoch {epoch+1}/{epochs} ---')
        
        train_profit = trainer.train_epoch(train_loader)  # 返回平均利润
        val_profit = trainer.validate(val_loader)
        
        train_profits.append(train_profit)
        val_profits.append(val_profit)
        
        trainer.scheduler.step(val_profit)  # 基于利润调整学习率
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        print(f'Train Profit: {train_profit:.2f}元 | Val Profit: {val_profit:.2f}元 | LR: {current_lr:.6f}')
        
        # 保存最佳模型（利润最高）
        if val_profit > best_val_profit:
            best_val_profit = val_profit
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_profit': train_profit,
                'val_profit': val_profit,
                'train_profits': train_profits,
                'val_profits': val_profits
            }, 'best_lottery_model.pth')
            print(f'*** 新的最佳模型已保存! 验证利润: {val_profit:.2f}元 ***')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= early_stop_patience:
            print(f'\n早停触发! {early_stop_patience} 个epoch没有改善.')
            break
    
    print(f"\n训练完成!")
    print(f"最佳验证利润: {best_val_profit:.2f}元")
    print(f"模型已保存为: best_lottery_model.pth")
    
    # 绘制训练历史
    plot_training_history(train_profits, val_profits)
    
    return train_profits, val_profits

if __name__ == "__main__":
    train_losses, val_losses = main()
