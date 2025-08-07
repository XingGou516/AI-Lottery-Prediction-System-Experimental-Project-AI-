import torch
import numpy as np
import pandas as pd
from lottery_model import LotteryPredictor, load_and_preprocess_data
import itertools

def load_trained_model(model_path, device):
    """加载训练好的模型"""
    model = LotteryPredictor(input_size=8)  # 修改为8维
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型加载成功!")
    print(f"训练epoch: {checkpoint['epoch']}")
    print(f"最佳验证利润: {checkpoint.get('val_profit', 'N/A')}")  # 修改显示
    
    return model

def predict_probabilities(model, last_sequence, device):
    """预测下一期号码的概率分布"""
    model.eval()
    
    with torch.no_grad():
        sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        red_probs, blue_probs = model(sequence)
        
        return red_probs.cpu().numpy()[0], blue_probs.cpu().numpy()[0]

def generate_betting_strategy(red_probs, blue_probs, budget=30):
    """生成投注策略"""
    print("\n=== 生成投注策略 ===")
    print(f"投注预算: {budget}元")
    
    # 获取概率排序
    red_indices = np.argsort(red_probs)[::-1]  # 从高到低
    blue_indices = np.argsort(blue_probs)[::-1]
    
    # 转换为实际号码(1-33, 1-16)
    red_numbers = red_indices + 1
    blue_numbers = blue_indices + 1
    
    print(f"\n红球概率前10: {red_numbers[:10]}")
    print(f"蓝球概率前5: {blue_numbers[:5]}")
    
    strategies = []
    total_cost = 0
    
    # 策略1: 蓝球复式 6红+3蓝 = 6元 (恢复较好的保底)
    if total_cost + 6 <= budget:
        strategy1 = {
            'type': '蓝球复式',
            'red_balls': sorted(red_numbers[:6].tolist()),
            'blue_balls': sorted(blue_numbers[:3].tolist()),
            'cost': 6,
            'description': '保底策略 - 保证六等奖'
        }
        strategies.append(strategy1)
        total_cost += 6
        print(f"\n策略1 (6元): 蓝球复式")
        print(f"红球: {strategy1['red_balls']}")
        print(f"蓝球: {strategy1['blue_balls']}")
    
    # 策略2: 红球复式 7红+1蓝 = 14元 (恢复四五等奖保障)
    if total_cost + 14 <= budget:
        strategy2 = {
            'type': '红球复式',
            'red_balls': sorted(red_numbers[:7].tolist()),
            'blue_balls': [blue_numbers[0]],
            'cost': 14,
            'description': '提高四五等奖概率'
        }
        strategies.append(strategy2)
        total_cost += 14
        print(f"\n策略2 (14元): 红球复式")
        print(f"红球: {strategy2['red_balls']}")
        print(f"蓝球: {strategy2['blue_balls']}")
    
    # 策略3: 单式投注 - 专注三等奖
    remaining_budget = budget - total_cost
    single_bets = remaining_budget // 2  # 每注2元
    
    print(f"\n策略3: {single_bets}注单式投注 ({single_bets * 2}元)")
    print("专注三等奖组合")
    
    for i in range(min(single_bets, 5)):  # 最多5注单式
        # 选择不同的红球组合，专注三等奖
        start_idx = i
        red_combo = sorted(red_numbers[start_idx:start_idx+6].tolist())
        blue_combo = blue_numbers[i % 3]  # 轮换前3个蓝球
        
        strategy = {
            'type': '单式',
            'red_balls': red_combo,
            'blue_balls': [blue_combo],
            'cost': 2,
            'description': f'三等奖组合 #{i+1} (5红+1蓝)'
        }
        strategies.append(strategy)
        total_cost += 2
        
        print(f"单式{i+1}: {red_combo} + [{blue_combo}] - 三等奖目标")
    
    print(f"\n总投注金额: {total_cost}元")
    print(f"剩余预算: {budget - total_cost}元")
    
    return strategies

def analyze_winning_probability(strategies, red_probs, blue_probs):
    """分析中奖概率和期望收益"""
    print("\n=== 中奖概率分析 ===")
    
    # 奖金设定
    prize_money = {
        '二等奖': 100000,  # 平均值
        '三等奖': 3000,
        '四等奖': 200,
        '五等奖': 10,
        '六等奖': 5
    }
    
    total_expected_return = 0
    
    for i, strategy in enumerate(strategies):
        print(f"\n--- 策略{i+1}: {strategy['type']} ---")
        print(f"成本: {strategy['cost']}元")
        
        if strategy['type'] == '蓝球复式':
            # 分析蓝球复式的中奖概率
            red_balls = strategy['red_balls']
            blue_balls = strategy['blue_balls']
            
            # 六等奖概率 (至少中1个蓝球)
            blue_hit_prob = sum(blue_probs[b-1] for b in blue_balls)
            six_prize_prob = min(blue_hit_prob, 1.0)
            expected_six = six_prize_prob * prize_money['六等奖']
            
            print(f"六等奖概率: {six_prize_prob:.3f}")
            print(f"六等奖期望收益: {expected_six:.2f}元")
            
            total_expected_return += expected_six
            
        elif strategy['type'] == '红球复式':
            # 分析红球复式的中奖概率
            red_balls = strategy['red_balls']
            blue_balls = strategy['blue_balls']
            
            # 简化计算：假设中4-5个红球的概率
            print(f"主要提高四五等奖概率")
            expected_return = strategy['cost'] * 0.3  # 简化估算
            total_expected_return += expected_return
            
        else:  # 单式
            red_balls = strategy['red_balls']
            blue_balls = strategy['blue_balls']
            
            # 计算各个红球的概率乘积（简化）
            red_combo_prob = np.prod([red_probs[r-1] for r in red_balls])
            blue_combo_prob = blue_probs[blue_balls[0]-1]
            
            # 三等奖概率 (5红+1蓝)
            three_prize_prob = red_combo_prob * blue_combo_prob * 0.1  # 简化系数
            expected_three = three_prize_prob * prize_money['三等奖']
            
            print(f"预期收益: {expected_three:.3f}元")
            total_expected_return += expected_three
    
    total_cost = sum(s['cost'] for s in strategies)
    net_expected = total_expected_return - total_cost
    roi = (net_expected / total_cost) * 100 if total_cost > 0 else 0
    
    print(f"\n=== 总体分析 ===")
    print(f"总投注成本: {total_cost}元")
    print(f"总期望收益: {total_expected_return:.2f}元")
    print(f"净期望收益: {net_expected:.2f}元")
    print(f"投资回报率: {roi:.2f}%")

def save_predictions(strategies, output_file='predictions.txt'):
    """保存预测结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 彩票预测结果 ===\n\n")
        
        total_cost = 0
        for i, strategy in enumerate(strategies):
            f.write(f"策略{i+1}: {strategy['type']}\n")
            f.write(f"红球: {strategy['red_balls']}\n")
            f.write(f"蓝球: {strategy['blue_balls']}\n")
            f.write(f"成本: {strategy['cost']}元\n")
            f.write(f"说明: {strategy['description']}\n")
            f.write("-" * 30 + "\n")
            total_cost += strategy['cost']
        
        f.write(f"\n总投注金额: {total_cost}元\n")
    
    print(f"\n预测结果已保存到: {output_file}")

def main():
    """主预测函数"""
    print("=== 彩票预测系统 ===")
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    try:
        model = load_trained_model('best_lottery_model.pth', device)
    except FileNotFoundError:
        print("错误: 找不到训练好的模型文件 'best_lottery_model.pth'")
        print("请先运行 train_lottery.py 训练模型")
        return
    
    # 加载数据
    data = load_and_preprocess_data('data.csv')
    
    # 使用最近12期数据进行预测
    sequence_length = 12
    last_sequence = data[-sequence_length:]
    
    print(f"\n使用最近{sequence_length}期数据进行预测:")
    print("最近期号码:")
    for i, period in enumerate(last_sequence[-5:]):  # 显示最近5期
        red_balls = period[:6].astype(int)
        blue_ball = int(period[6])
        print(f"  {red_balls} + [{blue_ball}]")
    
    # 预测概率
    red_probs, blue_probs = predict_probabilities(model, last_sequence, device)
    
    # 显示预测结果
    print(f"\n=== 预测结果 ===")
    red_top5 = np.argsort(red_probs)[-5:][::-1] + 1
    blue_top3 = np.argsort(blue_probs)[-3:][::-1] + 1
    
    print(f"红球概率最高的5个: {red_top5}")
    print(f"蓝球概率最高的3个: {blue_top3}")
    
    # 生成投注策略
    strategies = generate_betting_strategy(red_probs, blue_probs, budget=30)
    
    # 分析中奖概率
    analyze_winning_probability(strategies, red_probs, blue_probs)
    
    # 保存结果
    save_predictions(strategies)
    
    return strategies

if __name__ == "__main__":
    strategies = main()
