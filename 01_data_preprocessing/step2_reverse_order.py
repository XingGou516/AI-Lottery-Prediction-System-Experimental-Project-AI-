"""
第二步：调整数据顺序
- 将数据从旧到新（最新数据在前）调整为从新到旧（最新数据在后）
- 确保时间序列的正确性，便于LSTM学习
"""

import pandas as pd
import os

def reverse_csv_order(input_file, output_file=None):
    """
    将CSV文件中的数据从后往前重新排序
    
    参数:
    input_file: 输入的CSV文件路径
    output_file: 输出的CSV文件路径，如果为None则覆盖原文件
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：找不到文件 {input_file}")
            return False
        
        # 读取CSV文件，不使用列名
        df = pd.read_csv(input_file, header=None)
        
        print(f"成功读取文件：{input_file}")
        print(f"原始数据形状：{df.shape}")
        print("\n原始数据前5行（最新期）：")
        print(df.head())
        
        # 将数据行从后往前重新排序
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        
        # 确保个位数保留前置零
        for col in range(1, 8):  # 第二到第八列（索引1-7）
            if col < len(df_reversed.columns):
                # 转换为整数后再格式化为两位数字符串
                df_reversed[col] = df_reversed[col].astype(str).str.zfill(2)
        
        print("\n反转后数据前5行（最早期）：")
        print(df_reversed.head())
        print("\n反转后数据后5行（最新期）：")
        print(df_reversed.tail())
        
        # 如果没有指定输出文件，则覆盖原文件
        if output_file is None:
            output_file = input_file
        
        # 保存反转后的数据
        df_reversed.to_csv(output_file, index=False, header=False, encoding='utf-8-sig')
        
        print(f"\n成功保存到文件：{output_file}")
        print(f"反转后数据形状：{df_reversed.shape}")
        print("现在数据按时间顺序排列：从早期到最新期")
        
        return True
        
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")
        return False

def main():
    # 设置输入文件
    input_file = "../data.csv"
    
    print("=== 步骤2：调整数据时间顺序 ===")
    
    # 执行反转操作
    success = reverse_csv_order(input_file)
    
    if success:
        print("\n数据顺序调整完成！")
        print("数据现在按时间顺序排列，适合LSTM训练")
    else:
        print("\n数据顺序调整失败！")

if __name__ == "__main__":
    main()
