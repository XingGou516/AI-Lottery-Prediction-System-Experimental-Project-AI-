import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from lottery_model import LotteryDataset, LotteryPredictor, load_and_preprocess_data, analyze_data

class NoReplacementLotteryTrainer:
    """无放回抽取的彩票预测训练器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=15, factor=0.5
        )
        
    def calculate_combination_loss(self, red_logits, red_targets):
        """计算组合损失 - 考虑无放回约束"""
        batch_size = red_logits.shape[0]
        total_loss = 0
        
        for b in range(batch_size):
            # 创建目标向量：6个红球对应位置为1，其他为0
            target_vector = torch.zeros(33, device=self.device)
            for red_ball in red_targets[b]:
                target_vector[red_ball] = 1.0
            
            # 使用带logits的二元交叉熵，自动处理sigmoid
            loss = self.bce_criterion(red_logits[b], target_vector)
            total_loss += loss
            
        return total_loss / batch_size
    
    def sample_without_replacement(self, probs, k=6):
        """从概率分布中无放回采样k个元素"""
        batch_size = probs.shape[0]
        results = []
        
        for b in range(batch_size):
            prob_tensor = probs[b].clone()
            selected = []
            
            for _ in range(k):
                # 从当前概率分布中采样
                idx = torch.multinomial(prob_tensor, 1).item()
                selected.append(idx)
                # 将已选中的概率设为0，实现无放回
                prob_tensor[idx] = 0.0
                # 重新归一化
                if prob_tensor.sum() > 0:
                    prob_tensor = prob_tensor / prob_tensor.sum()
                else:
                    break
                    
            results.append(selected)
        
        return results
    
    def calculate_accuracy_no_replacement(self, dataloader):
        """考虑无放回约束的准确率计算"""
        self.model.eval()
        red_hits_total = 0
        blue_correct = 0
        total_samples = 0
        perfect_red_matches = 0
        perfect_all_matches = 0
        
        # 不同命中数的统计
        hit_counts = {i: 0 for i in range(7)}  # 0-6个红球命中
        
        with torch.no_grad():
            for sequences, red_targets, blue_targets in dataloader:
                sequences = sequences.to(self.device)
                red_targets = red_targets.to(self.device)
                blue_targets = blue_targets.to(self.device).squeeze()
                
                red_logits, blue_probs = self.model(sequences)
                red_probs = torch.softmax(red_logits, dim=-1)
                
                batch_size = red_probs.shape[0]
                for b in range(batch_size):
                    # 方法1：取概率最高的6个（可能有重复问题）
                    pred_red_top6 = torch.topk(red_probs[b], k=6).indices
                    
                    # 方法2：无放回采样（更符合实际）
                    pred_red_sampled = self.sample_without_replacement(red_probs[b:b+1], k=6)[0]
                    
                    actual_red_set = set(red_targets[b].cpu().numpy())
                    pred_red_set = set(pred_red_sampled)
                    
                    # 计算命中个数
                    red_hits = len(actual_red_set.intersection(pred_red_set))
                    red_hits_total += red_hits
                    hit_counts[red_hits] += 1
                    
                    # 蓝球预测
                    pred_blue = torch.argmax(blue_probs[b])
                    blue_hit = (pred_blue == blue_targets[b])
                    if blue_hit:
                        blue_correct += 1
                    
                    # 统计完美匹配
                    if red_hits == 6:
                        perfect_red_matches += 1
                        if blue_hit:
                            perfect_all_matches += 1
                
                total_samples += batch_size
        
        avg_red_hits = red_hits_total / total_samples
        blue_accuracy = blue_correct / total_samples
        perfect_red_rate = perfect_red_matches / total_samples
        perfect_all_rate = perfect_all_matches / total_samples
        
        return avg_red_hits, blue_accuracy, perfect_red_rate, perfect_all_rate, hit_counts
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            red_targets = red_targets.to(self.device)
            blue_targets = blue_targets.to(self.device).squeeze()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            red_logits, blue_probs = self.model(sequences)
            
            # 组合损失（考虑无放回）
            red_loss = self.calculate_combination_loss(red_logits, red_targets)
            
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
                print(f'  Batch {batch_idx:3d} | Red Loss: {red_loss.item():.4f} | Blue Loss: {blue_loss.item():.4f}')
                
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
                
                red_logits, blue_probs = self.model(sequences)
                
                red_loss = self.calculate_combination_loss(red_logits, red_targets)
                blue_loss = self.criterion(blue_probs, blue_targets)
                total_loss_batch = 0.8 * red_loss + 0.2 * blue_loss
                
                total_loss += total_loss_batch.item()
                
        return total_loss / len(dataloader)

def calculate_theoretical_probabilities():
    """计算理论概率"""
    # 红球：从33个中选6个，每个位置命中的概率
    # C(6,k) * C(27,6-k) / C(33,6)
    from math import comb
    
    total_combinations = comb(33, 6)
    print(f"\n=== 理论概率分析 ===")
    print(f"总的红球组合数: {total_combinations:,}")
    
    for k in range(7):
        if k <= 6:
            prob = comb(6, k) * comb(27, 6-k) / total_combinations
            print(f"命中{k}个红球的概率: {prob:.6f} ({prob*100:.4f}%)")
    
    print(f"蓝球命中概率: {1/16:.6f} ({100/16:.4f}%)")
    print(f"理论平均红球命中数: {6*6/33:.3f}")

def main():
    """实验2主函数 - 无放回约束"""
    print("=== 实验2：无放回抽取约束训练 ===")
    
    # 计算理论概率
    calculate_theoretical_probabilities()
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
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
    trainer = NoReplacementLotteryTrainer(model, device)
    
    # 训练模型
    print("\n开始训练（无放回约束）...")
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
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
        
        # 每10个epoch详细评估
        if epoch % 10 == 0:
            avg_red_hits, blue_acc, perfect_red_rate, perfect_all_rate, hit_counts = trainer.calculate_accuracy_no_replacement(val_loader)
            print(f'平均红球命中: {avg_red_hits:.3f}/6 | 蓝球准确率: {blue_acc:.3f}')
            print(f'红球全中率: {perfect_red_rate:.6f} | 全部命中率: {perfect_all_rate:.6f}')
            print(f'命中分布: {[f"{k}球:{v}" for k, v in hit_counts.items() if v > 0]}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'experiment2_no_replacement_model.pth')
            print(f'*** 最佳模型已保存! 验证损失: {val_loss:.4f} ***')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= early_stop_patience:
            print(f'\n早停触发! {early_stop_patience} 个epoch没有改善.')
            break
    
    print(f"\n实验2训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 最终详细评估
    print("\n=== 最终模型评估 ===")
    avg_red_hits, blue_acc, perfect_red_rate, perfect_all_rate, hit_counts = trainer.calculate_accuracy_no_replacement(val_loader)
    print(f"平均红球命中数: {avg_red_hits:.3f}/6 (理论随机: {6*6/33:.3f})")
    print(f"蓝球准确率: {blue_acc:.3f} (理论随机: {1/16:.3f})")
    print(f"红球全中率: {perfect_red_rate:.6f}")
    print(f"全部命中率: {perfect_all_rate:.6f}")
    
    print(f"\n红球命中分布:")
    total_samples = sum(hit_counts.values())
    for k in range(7):
        count = hit_counts[k]
        rate = count / total_samples if total_samples > 0 else 0
        print(f"  {k}个红球: {count} 次 ({rate:.3f})")
    
    return train_losses, val_losses

if __name__ == "__main__":
    train_losses, val_losses = main()
