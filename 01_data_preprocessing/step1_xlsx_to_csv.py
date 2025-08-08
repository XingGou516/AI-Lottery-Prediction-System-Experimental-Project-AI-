"""
第一步：Excel数据转换为CSV格式
- 处理原始xlsx文件
- 格式化号码（保证两位数格式）
- 输出标准CSV格式
"""

import pandas as pd
import os

def xlsx_to_csv(xlsx_file, csv_file):
    """
    将xlsx文件转换为csv格式
    xlsx文件结构：
    - 第一行：标题
    - 第二行：列名
    - 第三行开始：数据样本
    - 列结构：第一列为期号，第二到六列为红色球号，第七列为蓝色球号
    """
    try:
        # 读取xlsx文件，跳过前两行（标题和列名），不使用列名
        df = pd.read_excel(xlsx_file, header=None, skiprows=2)
        
        # 显示文件基本信息
        print(f"成功读取文件：{xlsx_file}")
        print(f"数据形状：{df.shape}")
        print("\n前5行数据：")
        print(df.head())
        
        # 处理数据格式，确保个位数保留前置零
        # 第二到七列为红色球号，第八列为蓝色球号，需要格式化为两位数
        for col in range(1, 8):  # 第二到第八列（索引1-7）
            if col < len(df.columns):
                df[col] = df[col].astype(int).apply(lambda x: f"{x:02d}")
        
        # 保存为csv文件，不包含列名和行索引
        df.to_csv(csv_file, index=False, header=False, encoding='utf-8-sig')
        print(f"\n成功转换为CSV文件：{csv_file}")
        
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误：{str(e)}")
        return False

def main():
    # 设置文件路径
    xlsx_file = "../data.xlsx"
    csv_file = "../data.csv"
    
    # 检查xlsx文件是否存在
    if not os.path.exists(xlsx_file):
        print(f"错误：找不到文件 {xlsx_file}")
        return
    
    print("=== 步骤1：Excel数据转换 ===")
    success = xlsx_to_csv(xlsx_file, csv_file)
    
    if success:
        print("\n转换完成！")
        
        # 显示转换后的csv文件信息
        try:
            df_csv = pd.read_csv(csv_file, header=None)
            print(f"\nCSV文件验证：")
            print(f"行数：{len(df_csv)}")
            print(f"列数：{len(df_csv.columns)}")
            print("\n转换后的前5行数据：")
            print(df_csv.head())
        except Exception as e:
            print(f"验证CSV文件时出现错误：{str(e)}")
    else:
        print("转换失败！")

if __name__ == "__main__":
    main()
