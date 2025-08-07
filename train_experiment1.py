import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lottery_model import LotteryDataset, LotteryPredictor, load_and_preprocess_data, analyze_data

class SimpleLotteryTrainer:
    """简化版彩票预测训练器 - 只使用交叉熵损失"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5  # 改回min，优化损失
        )
        
    def train_epoch(self, dataloader):
        """训练一个epoch - 简化版损失函数"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            red_targets = red_targets.to(self.device)
            blue_targets = blue_targets.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            red_probs, blue_probs = self.model(sequences)
            
            # 简化损失：只用交叉熵，不计算实际利润
            red_loss = 0
            for i in range(6):  # 6个红球
                red_loss += self.criterion(red_probs, red_targets[:, i])
            red_loss /= 6
            
            blue_loss = self.criterion(blue_probs, blue_targets)
            
            # 总损失：重点关注红球预测
            total_loss_batch = 0.7 * red_loss + 0.3 * blue_loss
            
            # 反向传播
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx:3d} | Loss: {total_loss_batch.item():.4f}')
                
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """验证模型 - 计算交叉熵损失"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_probs, blue_probs = self.model(sequences)
                
                # 计算损失
                red_loss = 0
                for i in range(6):
                    red_loss += self.criterion(red_probs, red_targets[:, i])
                red_loss /= 6
                
                blue_loss = self.criterion(blue_probs, blue_targets)
                total_loss_batch = 0.7 * red_loss + 0.3 * blue_loss
                
                total_loss += total_loss_batch.item()
                
        return total_loss / len(dataloader)
    
    def calculate_accuracy(self, dataloader):
        """计算预测准确率"""
        self.model.eval()
        red_correct = 0
        blue_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_probs, blue_probs = self.model(sequences)
                
                # 红球准确率（取概率最高的6个）
                batch_size = red_probs.shape[0]
                for b in range(batch_size):
                    # 预测的前6个红球
                    pred_red = torch.topk(red_probs[b], k=6).indices
                    actual_red = set(red_targets[b].cpu().numpy())
                    pred_red_set = set(pred_red.cpu().numpy())
                    
                    # 计算命中个数
                    red_hits = len(actual_red.intersection(pred_red_set))
                    red_correct += red_hits
                    
                    # 蓝球准确率
                    pred_blue = torch.argmax(blue_probs[b])
                    if pred_blue == blue_targets[b]:
                        blue_correct += 1
                
                total_samples += batch_size
        
        red_accuracy = red_correct / (total_samples * 6)  # 平均每个红球位置的准确率
        blue_accuracy = blue_correct / total_samples
        
        return red_accuracy, blue_accuracy

def plot_training_history(train_losses, val_losses):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss (Simple)')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-50:], label='Training Loss (Last 50 epochs)')
    plt.plot(val_losses[-50:], label='Validation Loss (Last 50 epochs)')
    plt.title('Model Loss (Recent)')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment1_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """实验1主函数"""
    print("=== 实验1：简化损失函数训练 ===")
    
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
    
    # 按时间顺序分割
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
    trainer = SimpleLotteryTrainer(model, device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练模型
    print("\n开始训练（简化损失函数）...")
    epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    early_stop_patience = 20
    
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
            red_acc, blue_acc = trainer.calculate_accuracy(val_loader)
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
            print(f'Red Accuracy: {red_acc:.3f} | Blue Accuracy: {blue_acc:.3f}')
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
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'experiment1_best_model.pth')
            print(f'*** 新的最佳模型已保存! 验证损失: {val_loss:.4f} ***')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= early_stop_patience:
            print(f'\n早停触发! {early_stop_patience} 个epoch没有改善.')
            break
    
    print(f"\n实验1训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存为: experiment1_best_model.pth")
    
    # 最终准确率评估
    print("\n=== 最终模型评估 ===")
    red_acc, blue_acc = trainer.calculate_accuracy(val_loader)
    print(f"红球平均准确率: {red_acc:.3f}")
    print(f"蓝球准确率: {blue_acc:.3f}")
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses)
    
    return train_losses, val_losses

if __name__ == "__main__":
    train_losses, val_losses = main()
