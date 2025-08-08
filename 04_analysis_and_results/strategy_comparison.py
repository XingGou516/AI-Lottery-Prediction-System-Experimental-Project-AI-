"""
策略对比实验
- 测试不同预算下的投注策略效果
- 对比保守、极简、自适应、频率等多种策略
- 结果：所有策略都显示负收益，6元预算表现最佳
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_models'))
from lottery_model import LotteryPredictor, load_and_preprocess_data

def load_best_model():
    """加载最好的预测模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LotteryPredictor(input_size=8)
    
    # 尝试加载不同实验的最佳模型
    try:
        checkpoint = torch.load('../saved_models/experiment1_5_best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("加载实验1.5模型成功")
    except:
        try:
            checkpoint = torch.load('../saved_models/experiment2_no_replacement_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("加载实验2模型成功")
        except:
            print("警告：无法加载预训练模型，使用随机初始化")
    
    model.to(device)
    model.eval()
    return model, device

def strategy_conservative(red_probs, blue_probs, budget=10):
    """保守策略：10元预算"""
    red_indices = np.argsort(red_probs)[::-1] + 1
    blue_indices = np.argsort(blue_probs)[::-1] + 1
    
    strategies = []
    
    # 1注蓝球复式 6红+2蓝 = 4元
    strategies.append({
        'type': 'blue_complex',
        'red_balls': set(red_indices[:6]),
        'blue_balls': set(blue_indices[:2]),
        'cost': 4
    })
    
    # 3注单式 = 6元
    for i in range(3):
        strategies.append({
            'type': 'single',
            'red_balls': set(red_indices[i*2:(i*2)+6]),
            'blue_balls': set([blue_indices[i % 2]]),
            'cost': 2
        })
    
    return strategies

def strategy_minimal(red_probs, blue_probs, budget=6):
    """极简策略：6元预算"""
    red_indices = np.argsort(red_probs)[::-1] + 1
    blue_indices = np.argsort(blue_probs)[::-1] + 1
    
    strategies = []
    
    # 3注单式 = 6元，专注三等奖
    for i in range(3):
        strategies.append({
            'type': 'single',
            'red_balls': set(red_indices[i*2:(i*2)+6]),
            'blue_balls': set([blue_indices[i % 2]]),
            'cost': 2
        })
    
    return strategies

def strategy_adaptive(red_probs, blue_probs, budget=16):
    """自适应策略：16元预算"""
    red_indices = np.argsort(red_probs)[::-1] + 1
    blue_indices = np.argsort(blue_probs)[::-1] + 1
    
    # 计算概率集中度
    red_entropy = -np.sum(red_probs * np.log(red_probs + 1e-8))
    blue_entropy = -np.sum(blue_probs * np.log(blue_probs + 1e-8))
    
    strategies = []
    
    if red_entropy < 3.0:  # 红球概率较集中
        # 偏向复式投注
        strategies.append({
            'type': 'red_complex',
            'red_balls': set(red_indices[:7]),
            'blue_balls': set(blue_indices[:1]),
            'cost': 14
        })
        # 补充单式
        strategies.append({
            'type': 'single',
            'red_balls': set(red_indices[:6]),
            'blue_balls': set([blue_indices[0]]),
            'cost': 2
        })
    else:  # 红球概率分散
        # 偏向单式投注
        for i in range(8):
            strategies.append({
                'type': 'single',
                'red_balls': set(red_indices[i:(i+6)]),
                'blue_balls': set([blue_indices[i % 3]]),
                'cost': 2
            })
    
    return strategies

def strategy_frequency_based(red_probs, blue_probs, historical_data, budget=12):
    """基于历史频率的策略：12元预算"""
    # 计算最近10期的热号
    recent_data = historical_data[-10:]
    red_freq = {}
    blue_freq = {}
    
    for period in recent_data:
        # period[1:7]是红球，period[7]是蓝球
        for red_ball in period[1:7]:
            red_freq[int(red_ball)] = red_freq.get(int(red_ball), 0) + 1
        blue_ball = int(period[7])
        blue_freq[blue_ball] = blue_freq.get(blue_ball, 0) + 1
    
    # 结合模型预测和历史频率
    red_indices = np.argsort(red_probs)[::-1] + 1
    blue_indices = np.argsort(blue_probs)[::-1] + 1
    
    # 热号调整
    hot_red = sorted(red_freq.keys(), key=lambda x: red_freq.get(x, 0), reverse=True)[:10]
    hot_blue = sorted(blue_freq.keys(), key=lambda x: blue_freq.get(x, 0), reverse=True)[:3]
    
    # 混合策略：50%模型预测 + 50%历史热号
    mixed_red = []
    for i in range(10):
        if i < 5:
            mixed_red.append(red_indices[i])
        else:
            mixed_red.append(hot_red[i-5] if i-5 < len(hot_red) else red_indices[i])
    
    strategies = []
    
    # 蓝球复式：使用热号
    strategies.append({
        'type': 'blue_complex',
        'red_balls': set(mixed_red[:6]),
        'blue_balls': set(hot_blue[:2] if len(hot_blue) >= 2 else blue_indices[:2]),
        'cost': 4
    })
    
    # 单式：混合策略
    for i in range(4):
        strategies.append({
            'type': 'single',
            'red_balls': set(mixed_red[i:(i+6)]),
            'blue_balls': set([hot_blue[i % len(hot_blue)] if hot_blue else blue_indices[i % 3]]),
            'cost': 2
        })
    
    return strategies

def calculate_win_amount(strategy, actual_red_set, actual_blue_num):
    """计算中奖金额"""
    prize_money = {2: 50000, 3: 3000, 4: 200, 5: 10, 6: 5}
    
    red_hits = len(actual_red_set.intersection(strategy['red_balls']))
    blue_hit = actual_blue_num in strategy['blue_balls']
    
    if strategy['type'] == 'single':
        if red_hits == 6 and blue_hit:
            return prize_money[2]
        elif red_hits == 6:
            return prize_money[3]
        elif red_hits == 5 and blue_hit:
            return prize_money[4]
        elif red_hits == 5 or (red_hits == 4 and blue_hit):
            return prize_money[5]
        elif red_hits == 4 or (red_hits == 3 and blue_hit):
            return prize_money[6]
            
    elif strategy['type'] == 'blue_complex':
        if red_hits == 6 and blue_hit:
            return prize_money[2]
        elif red_hits == 6:
            return prize_money[3] * len(strategy['blue_balls'])
        elif red_hits == 5 and blue_hit:
            return prize_money[4] * len(strategy['blue_balls'])
        elif red_hits == 5:
            return prize_money[5] * len(strategy['blue_balls'])
        elif red_hits == 4 and blue_hit:
            return prize_money[5] * len(strategy['blue_balls'])
        elif red_hits == 4:
            return prize_money[6] * len(strategy['blue_balls'])
            
    elif strategy['type'] == 'red_complex':
        if red_hits == 6 and blue_hit:
            return prize_money[2]
        elif red_hits == 6:
            return prize_money[3]
        elif red_hits == 5 and blue_hit:
            return prize_money[4]
        elif red_hits == 5:
            return prize_money[5]
        elif red_hits == 4 and blue_hit:
            return prize_money[5]
        elif red_hits == 4:
            return prize_money[6]
    
    return 0

def test_strategy(strategy_func, model, device, test_data, strategy_name, budget):
    """测试投注策略"""
    print(f"\n=== 测试策略: {strategy_name} (预算:{budget}元) ===")
    
    total_cost = 0
    total_win = 0
    total_periods = 0
    win_periods = 0
    
    profits = []
    
    # 使用最后100期进行测试
    test_periods = test_data[-100:]
    
    for i in range(len(test_periods) - 12):
        # 准备输入序列
        sequence = test_periods[i:i+12]
        sequence_normalized = sequence.copy()
        # 期号归一化
        period_min, period_max = test_data[:, 0].min(), test_data[:, 0].max()
        sequence_normalized[:, 0] = (sequence[:, 0] - period_min) / (period_max - period_min)
        
        # 目标期（下一期）
        target_period = test_periods[i+12]
        actual_red_set = set(target_period[1:7].astype(int))
        actual_blue_num = int(target_period[7])
        
        # 模型预测
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(device)
            red_probs, blue_probs = model(sequence_tensor)
            red_probs = red_probs[0].cpu().numpy()
            blue_probs = blue_probs[0].cpu().numpy()
        
        # 生成投注策略
        if strategy_name == "Frequency":
            strategies = strategy_func(red_probs, blue_probs, test_data, budget)
        else:
            strategies = strategy_func(red_probs, blue_probs, budget)
        
        # 计算收益
        period_cost = sum(s['cost'] for s in strategies)
        period_win = sum(calculate_win_amount(s, actual_red_set, actual_blue_num) for s in strategies)
        period_profit = period_win - period_cost
        
        total_cost += period_cost
        total_win += period_win
        profits.append(period_profit)
        total_periods += 1
        
        if period_profit > 0:
            win_periods += 1
    
    # 统计结果
    avg_profit = (total_win - total_cost) / total_periods
    win_rate = win_periods / total_periods
    roi = ((total_win - total_cost) / total_cost) * 100
    
    print(f"测试期数: {total_periods}")
    print(f"总投入: {total_cost}元")
    print(f"总收入: {total_win}元")
    print(f"总利润: {total_win - total_cost}元")
    print(f"平均每期利润: {avg_profit:.2f}元")
    print(f"盈利期数比例: {win_rate:.1%}")
    print(f"投资回报率: {roi:.2f}%")
    print(f"最大单期盈利: {max(profits):.2f}元")
    print(f"最大单期亏损: {min(profits):.2f}元")
    
    return {
        'strategy_name': strategy_name,
        'budget': budget,
        'total_periods': total_periods,
        'total_cost': total_cost,
        'total_win': total_win,
        'total_profit': total_win - total_cost,
        'avg_profit': avg_profit,
        'win_rate': win_rate,
        'roi': roi,
        'max_profit': max(profits),
        'max_loss': min(profits),
        'profits': profits
    }

def plot_strategy_comparison(results):
    """绘制策略对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    strategies = [r['strategy_name'] for r in results]
    budgets = [r['budget'] for r in results]
    rois = [r['roi'] for r in results]
    avg_profits = [r['avg_profit'] for r in results]
    win_rates = [r['win_rate'] * 100 for r in results]
    
    # ROI对比
    axes[0, 0].bar(strategies, rois, color=['blue', 'green', 'red', 'orange'])
    axes[0, 0].set_title('ROI Comparison')
    axes[0, 0].set_ylabel('ROI (%)')
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 平均利润对比
    axes[0, 1].bar(strategies, avg_profits, color=['blue', 'green', 'red', 'orange'])
    axes[0, 1].set_title('Average Profit Comparison')
    axes[0, 1].set_ylabel('Average Profit (Yuan)')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 胜率对比
    axes[1, 0].bar(strategies, win_rates, color=['blue', 'green', 'red', 'orange'])
    axes[1, 0].set_title('Win Rate Comparison')
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 预算vs收益散点图
    axes[1, 1].scatter(budgets, rois, s=100, c=['blue', 'green', 'red', 'orange'])
    for i, strategy in enumerate(strategies):
        axes[1, 1].annotate(strategy, (budgets[i], rois[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    axes[1, 1].set_title('Budget vs ROI')
    axes[1, 1].set_xlabel('Budget (Yuan)')
    axes[1, 1].set_ylabel('ROI (%)')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../strategy_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """策略对比实验"""
    print("=== 投注策略对比实验 ===")
    print("目标：找出最优投注策略和预算配置")
    
    # 加载模型和数据
    model, device = load_best_model()
    data = load_and_preprocess_data('../data.csv')
    
    print(f"测试数据总期数: {len(data)}")
    
    # 定义策略
    strategies_to_test = [
        (strategy_minimal, "Minimal", 6),
        (strategy_conservative, "Conservative", 10),
        (strategy_frequency_based, "Frequency", 12),
        (strategy_adaptive, "Adaptive", 16),
    ]
    
    # 测试所有策略
    results = []
    for strategy_func, strategy_name, budget in strategies_to_test:
        result = test_strategy(strategy_func, model, device, data, strategy_name, budget)
        results.append(result)
    
    # 总结对比
    print("\n" + "="*60)
    print("策略对比总结")
    print("="*60)
    print(f"{'策略名称':<12} {'预算':<6} {'ROI':<8} {'平均利润':<10} {'胜率':<8}")
    print("-"*60)
    
    for result in results:
        print(f"{result['strategy_name']:<12} {result['budget']:<6}元 "
              f"{result['roi']:<8.1f}% {result['avg_profit']:<10.2f}元 "
              f"{result['win_rate']:<8.1%}")
    
    # 找出最佳策略
    best_roi = max(results, key=lambda x: x['roi'])
    best_profit = max(results, key=lambda x: x['avg_profit'])
    best_winrate = max(results, key=lambda x: x['win_rate'])
    
    print("\n" + "="*60)
    print("最佳策略分析")
    print("="*60)
    print(f"最高ROI: {best_roi['strategy_name']} ({best_roi['roi']:.1f}%)")
    print(f"最高平均利润: {best_profit['strategy_name']} ({best_profit['avg_profit']:.2f}元)")
    print(f"最高胜率: {best_winrate['strategy_name']} ({best_winrate['win_rate']:.1%})")
    
    # 绘制对比图
    plot_strategy_comparison(results)
    
    print("\n=== 策略对比实验结论 ===")
    print("1. 所有策略都显示负预期收益")
    print("2. 低预算策略相对表现更好")
    print("3. 预算增加并不带来收益提升")
    print("4. 彩票预测的根本困难得到验证")
    
    return results

if __name__ == "__main__":
    results = main()
