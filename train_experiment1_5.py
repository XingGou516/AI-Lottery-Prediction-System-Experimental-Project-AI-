import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lottery_model import LotteryDataset, LotteryPredictor, load_and_preprocess_data, analyze_data

class ImprovedLotteryTrainer:
    """改进版训练器 - 修复损失函数逻辑"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.5
        )
        
    def calculate_red_ball_loss(self, red_probs, red_targets):
        """改进的红球损失计算"""
        batch_size = red_probs.shape[0]
        total_loss = 0
        
        # 方法1：把6个红球看作一个多标签分类问题
        for b in range(batch_size):
            # 创建目标向量：命中的红球位置为1，其他为0
            target_vector = torch.zeros(33, device=self.device)
            for red_ball in red_targets[b]:
                target_vector[red_ball] = 1.0
            
            # 使用二元交叉熵损失
            loss = nn.functional.binary_cross_entropy(
                torch.sigmoid(red_probs[b]), target_vector
            )
            total_loss += loss
            
        return total_loss / batch_size
    
    def train_epoch(self, dataloader):
        """训练一个epoch - 改进版损失函数"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            red_targets = red_targets.to(self.device)
            blue_targets = blue_targets.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            red_probs, blue_probs = self.model(sequences)
            
            # 改进的红球损失计算
            red_loss = self.calculate_red_ball_loss(red_probs, red_targets)
            
            # 蓝球损失
            blue_loss = self.criterion(blue_probs, blue_targets)
            
            # 总损失
            total_loss_batch = 0.8 * red_loss + 0.2 * blue_loss
            
            # 反向传播
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx:3d} | Red Loss: {red_loss.item():.4f} | Blue Loss: {blue_loss.item():.4f} | Total: {total_loss_batch.item():.4f}')
                
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_probs, blue_probs = self.model(sequences)
                
                red_loss = self.calculate_red_ball_loss(red_probs, red_targets)
                blue_loss = self.criterion(blue_probs, blue_targets)
                total_loss_batch = 0.8 * red_loss + 0.2 * blue_loss
                
                total_loss += total_loss_batch.item()
                
        return total_loss / len(dataloader)
    
    def calculate_accuracy_improved(self, dataloader):
        """改进的准确率计算"""
        self.model.eval()
        red_hits_total = 0
        blue_correct = 0
        total_samples = 0
        perfect_matches = 0
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_probs, blue_probs = self.model(sequences)
                
                batch_size = red_probs.shape[0]
                for b in range(batch_size):
                    # 预测的前6个红球
                    pred_red_indices = torch.topk(red_probs[b], k=6).indices
                    actual_red_set = set(red_targets[b].cpu().numpy())
                    pred_red_set = set(pred_red_indices.cpu().numpy())
                    
                    # 计算命中个数
                    red_hits = len(actual_red_set.intersection(pred_red_set))
                    red_hits_total += red_hits
                    
                    # 蓝球预测
                    pred_blue = torch.argmax(blue_probs[b])
                    blue_hit = (pred_blue == blue_targets[b])
                    if blue_hit:
                        blue_correct += 1
                    
                    # 完美匹配（6红+1蓝全中）
                    if red_hits == 6 and blue_hit:
                        perfect_matches += 1
                
                total_samples += batch_size
        
        avg_red_hits = red_hits_total / total_samples  # 平均每期命中红球数
        blue_accuracy = blue_correct / total_samples
        perfect_rate = perfect_matches / total_samples
        
        return avg_red_hits, blue_accuracy, perfect_rate

def main():
    """实验1.5主函数 - 改进版损失函数"""
    print("=== 实验1.5：改进版损失函数训练 ===")
    
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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = LotteryPredictor(input_size=8)
    trainer = ImprovedLotteryTrainer(model, device)
    
    # 训练模型
    print("\n开始训练（改进版损失函数）...")
    epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    early_stop_patience = 25
    
    for epoch in range(epochs):
        print(f'\n--- Epoch {epoch+1}/{epochs} ---')
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        trainer.scheduler.step(val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # 每10个epoch计算一次准确率
        if epoch % 10 == 0:
            avg_red_hits, blue_acc, perfect_rate = trainer.calculate_accuracy_improved(val_loader)
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
            print(f'平均红球命中: {avg_red_hits:.2f}/6 | 蓝球准确率: {blue_acc:.3f} | 完美匹配率: {perfect_rate:.6f}')
        else:
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'experiment1_5_best_model.pth')
            print(f'*** 最佳模型已保存! 验证损失: {val_loss:.4f} ***')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= early_stop_patience:
            print(f'\n早停触发! {early_stop_patience} 个epoch没有改善.')
            break
    
    print(f"\n实验1.5训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 最终评估
    print("\n=== 最终模型评估 ===")
    avg_red_hits, blue_acc, perfect_rate = trainer.calculate_accuracy_improved(val_loader)
    print(f"平均红球命中数: {avg_red_hits:.2f}/6")
    print(f"蓝球准确率: {blue_acc:.3f}")
    print(f"完美匹配率: {perfect_rate:.6f}")
    print(f"理论随机红球命中: {6*6/33:.2f}/6")
    print(f"理论随机蓝球准确率: {1/16:.3f}")
    
    return train_losses, val_losses

if __name__ == "__main__":
    train_losses, val_losses = main()
