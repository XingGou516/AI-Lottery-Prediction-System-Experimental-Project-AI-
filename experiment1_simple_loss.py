# 实验1：简化损失函数
# 在train_lottery.py中的train_epoch函数里修改：

def train_epoch_simple(self, dataloader):
    """简化版训练 - 只用交叉熵损失"""
    self.model.train()
    total_loss = 0
    
    for batch_idx, (sequences, red_targets, blue_targets) in enumerate(dataloader):
        sequences = sequences.to(self.device)
        red_targets = red_targets.to(self.device)
        blue_targets = blue_targets.to(self.device).squeeze()
        
        self.optimizer.zero_grad()
        
        red_probs, blue_probs = self.model(sequences)
        
        # 简化损失：只用交叉熵
        red_loss = 0
        for i in range(6):
            red_loss += self.criterion(red_probs, red_targets[:, i])
        red_loss /= 6
        blue_loss = self.criterion(blue_probs, blue_targets)
        
        total_loss_batch = 0.7 * red_loss + 0.3 * blue_loss
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        total_loss += total_loss_batch.item()
        
    return total_loss / len(dataloader)
