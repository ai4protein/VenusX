import os
import random
from math import floor
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from math import floor

def split_csv(input_file, output_dir, train_file="train.csv", valid_file="valid.csv", test_file="test.csv"):
    """
    Split a single CSV file into training, validation, and test datasets in an 8:1:1 ratio.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the output CSV files (train.csv, valid.csv, test.csv).
        train_file (str): Name of the training output file. Default is 'train.csv'.
        valid_file (str): Name of the validation output file. Default is 'valid.csv'.
        test_file (str): Name of the test output file. Default is 'test.csv'.
    """
    # 1. Check if the input file exists and is valid
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return
    
    if os.path.getsize(input_file) == 0:
        print(f"Error: {input_file} is empty.")
        return

    # 2. Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return

    # 3. Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Split the data into 8:1:1 (train:valid:test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 5. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 6. Save the split datasets to CSV files
    train_df.to_csv(os.path.join(output_dir, train_file), index=False)
    valid_df.to_csv(os.path.join(output_dir, valid_file), index=False)
    test_df.to_csv(os.path.join(output_dir, test_file), index=False)

    print(f"Done! Files saved in: {output_dir}")

def merge_and_split_csv(root_dir, output_dir, file_suffix="merged.csv"):
    """
    合并所有子文件夹下的 CSV 文件，并按照 8:1:1 划分数据集。
    """
    # 1. 收集所有子文件夹下的 CSV 文件
    all_dfs = []
    for subfolder in tqdm(os.listdir(root_dir), desc="Processing subfolders"):
        sub_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(sub_path):  # 确保是文件夹
            for file in os.listdir(sub_path):
                if file.endswith(file_suffix):
                    file_path = os.path.join(sub_path, file)
                    if not os.path.exists(file_path):  # 检查文件是否存在
                        print(f"Warning: {file_path} does not exist, skipping...")
                        continue
                    if os.path.getsize(file_path) == 0:  # 检查文件是否为空
                        print(f"Warning: {file_path} is empty, skipping...")
                        continue
                    try:
                        df = pd.read_csv(file_path)  # 尝试读取文件
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    # 如果没有有效的 CSV 文件，直接返回
    if not all_dfs:
        print("No valid CSV files found. Exiting...")
        return

    # 2. 合并所有 CSV 数据
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 随机打乱

    # 3. 按照 8:1:1 划分数据集
    train_df, temp_df = train_test_split(full_df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 4. 保存为新的 CSV 文件
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Done! Files saved in:", output_dir)

def split_by_family_folders(root_dir, output_dir, seed=42):
    """
    按家族文件夹划分数据集，并保存为 CSV 文件。
    """
    random.seed(seed)
    
    # 1. 获取所有蛋白质家族文件夹（子文件夹）
    all_families = [
        f for f in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, f))
    ]
    random.shuffle(all_families)  # 随机打乱家族顺序

    # 2. 计算划分数量
    total = len(all_families)
    train_cutoff = floor(0.8 * total)
    valid_cutoff = floor(0.9 * total)

    train_folders = all_families[:train_cutoff]
    valid_folders = all_families[train_cutoff:valid_cutoff]
    test_folders = all_families[valid_cutoff:]

    # 3. 加载并合并每部分的 CSV 文件
    def load_csvs(folders):
        dfs = []
        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):  # 检查文件夹是否存在
                print(f"Warning: Folder {folder_path} does not exist, skipping...")
                continue
            for file in os.listdir(folder_path):
                if file.endswith("merged.csv"):  # 只处理 merged.csv 文件
                    file_path = os.path.join(folder_path, file)
                    if not os.path.exists(file_path):  # 检查文件是否存在
                        print(f"Warning: {file_path} does not exist, skipping...")
                        continue
                    if os.path.getsize(file_path) == 0:  # 检查文件是否为空
                        print(f"Warning: {file_path} is empty, skipping...")
                        continue
                    try:
                        df = pd.read_csv(file_path)  # 尝试读取文件
                        dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()  # 返回空 DataFrame

    train_df = load_csvs(train_folders)
    valid_df = load_csvs(valid_folders)
    test_df = load_csvs(test_folders)

    # 4. 保存为 CSV
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("跨家族划分完成！共 {} 个家族：训练集 {}，验证集 {}，测试集 {}".format(
        total, len(train_folders), len(valid_folders), len(test_folders)
    ))

def split_by_family_size_group_2(root_dir, output_dir, seed=3407, quantile=0.5):
    random.seed(seed)
    error_num = 0
    # 1. 统计家族大小
    family_sizes = {}
    family_files = {}
    for family in os.listdir(root_dir):
        folder = os.path.join(root_dir, family)
        if not os.path.isdir(folder):
            print(f"Warning: {folder} is not a directory, skipping...")
        for f in os.listdir(folder):
            if f.endswith("merged.csv"):
                path = os.path.join(folder, f)
                try:
                    df = pd.read_csv(path)
                    family_sizes[family] = len(df)
                    family_files[family] = path
                except:
                    error_num += 1
                break
    print(os.listdir(root_dir)-family_sizes.keys())
    total_families = len(family_sizes)
    print(f"总家族数: {total_families}；读取失败: {error_num}；有效家族数: {total_families - error_num}")
    if total_families == 0:
        raise ValueError("未找到任何有效的 merged.csv")

    # 2. 按分位数二分
    sizes = list(family_sizes.values())
    threshold = pd.Series(sizes).quantile(quantile)
    large = [fam for fam, sz in family_sizes.items() if sz >= threshold]
    small = [fam for fam, sz in family_sizes.items() if sz <  threshold]

    # 检查分组是否完整
    assert len(large) + len(small) == total_families, \
           f"分组后家族数不对：{len(large)} + {len(small)} != {total_families}"

    # 3. 组内切分函数
    def split_group(families):
        random.shuffle(families)
        n = len(families)
        t_cut = floor(0.8 * n)
        v_cut = floor(0.9 * n)
        tr = families[:t_cut]
        va = families[t_cut:v_cut]
        te = families[v_cut:]
        # 检查
        assert len(tr) + len(va) + len(te) == n, \
               f"组内切分出错：{len(tr)}+{len(va)}+{len(te)} != {n}"
        return tr, va, te

    # 4. 切分大/小组
    L_train, L_valid, L_test = split_group(large)
    S_train, S_valid, S_test = split_group(small)

    # 5. 合并并检查无重叠、无丢失
    train_fams = L_train + S_train
    valid_fams = L_valid + S_valid
    test_fams  = L_test  + S_test

    all_assigned = set(train_fams) | set(valid_fams) | set(test_fams)
    assert len(all_assigned) == total_families, \
           f"分配后家族数不对：{len(all_assigned)} != {total_families}"
    assert set(train_fams).isdisjoint(valid_fams)
    assert set(train_fams).isdisjoint(test_fams)
    assert set(valid_fams).isdisjoint(test_fams)

    # 6. 合并 CSV 并保存
    def merge_csv(families):
        dfs = []
        for fam in families:
            dfs.append(pd.read_csv(family_files[fam]))
        return pd.concat(dfs, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    merge_csv(train_fams).to_csv(os.path.join(output_dir, "train.csv"), index=False)
    merge_csv(valid_fams).to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    merge_csv(test_fams).to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # 7. 打印
    print("分组后 8:1:1 划分完成！")
    print(f"原始家族数: {total_families}；大组: {len(large)}；小组: {len(small)}")
    print(f"训练: {len(train_fams)} 家族；验证: {len(valid_fams)}；测试: {len(test_fams)}")
    print("如果断言都通过，则家族总数与 CSV 样本数都应正确。")

def count_csv_rows(root_dir):
    """
    统计指定路径下所有 CSV 文件的数据条数。
    
    参数:
        root_dir (str): 根目录路径。
        
    返回:
        dict: 每个 CSV 文件的路径及其对应的行数（不包括表头）。
    """
    csv_stats = {}  # 用于存储每个 CSV 文件的路径和行数

    # 遍历根目录及其子目录
    for folder_path, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv"):  # 只处理 CSV 文件
                file_path = os.path.join(folder_path, file)
                
                # 检查文件是否为空
                if os.path.getsize(file_path) == 0:
                    print(f"Warning: {file_path} is empty, skipping...")
                    csv_stats[file_path] = 0
                    continue
                
                # 尝试读取 CSV 文件并统计行数
                try:
                    df = pd.read_csv(file_path)
                    row_count = len(df)  # 获取数据行数（不包括表头）
                    csv_stats[file_path] = row_count
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    csv_stats[file_path] = 0  # 如果读取失败，记录为 0 行

    # 打印统计结果
    total_rows = sum(csv_stats.values())
    print(f"Total CSV files found: {len(csv_stats)}")
    print(f"Total rows across all CSV files: {total_rows}")
    
    return csv_stats

def process_interpro_labels(csv_path):
    """
    读取CSV文件，根据interpro_id分配从0开始的编号，新增interpro_label列，
    并保存回原文件，同时输出类别总数。

    参数:
        csv_path (str): CSV文件路径

    返回:
        int: interpro类别总数
    """
    # 读取CSV
    df = pd.read_csv(csv_path)

    # 获取唯一interpro_id并分配编号
    unique_ids = sorted(df['interpro_id'].unique())
    id_to_label = {interpro_id: idx for idx, interpro_id in enumerate(unique_ids)}

    # 添加新列
    df['interpro_label'] = df['interpro_id'].map(id_to_label)

    # 保存到原文件
    df.to_csv(csv_path, index=False)

    # 输出并返回类别数
    num_classes = len(unique_ids)
    print(f"Number of interpro_id classes: {num_classes}")

    return num_classes

def add_interpro_labels_from_master(data_dir, master_csv_path):
    """
    根据 master 文件中的 interpro_id → interpro_label 映射，
    将标签添加到 data_dir 中的 train.csv、valid.csv 和 test.csv 文件中。

    参数:
        data_dir (str): 包含 train/valid/test.csv 的文件夹路径
        master_csv_path (str): 包含 interpro_id 和 interpro_label 的主CSV文件路径
    """
    # 读取主文件，建立映射字典
    master_df = pd.read_csv(master_csv_path)
    label_map = dict(zip(master_df['interpro_id'], master_df['interpro_label']))

    # 处理三个数据集文件
    for split_name in ['train.csv', 'valid.csv', 'test.csv']:
        file_path = os.path.join(data_dir, split_name)
        if not os.path.isfile(file_path):
            print(f"Warning: {split_name} not found in {data_dir}, skipping.")
            continue

        df = pd.read_csv(file_path)

        # 添加 interpro_label 列
        df['interpro_label'] = df['interpro_id'].map(label_map)

        # 保存并覆盖原文件
        df.to_csv(file_path, index=False)
        print(f"✓ {split_name} updated with interpro_label.")

if __name__ == "__main__":
    # 1. 跨家族划分
    # split_by_family_size_group_2("/home/tanyang/workspace/VenusScope/data/interpro_2703/domain", 
    #                         "/home/tanyang/workspace/VenusScope/data/token_cls_dataset/cross_family/domain")

    # 2. 合并并划分数据集
    # merge_and_split_csv("/home/tanyang/workspace/VenusScope/data/interpro_2703/domain", 
    #                     "/home/tanyang/workspace/VenusScope/data/token_cls_dataset/cross_family/domain/sim_70",
    #                     "merged_70.csv")

    # split_csv(
    #     "/home/tanyang/workspace/VenusScope/data/binding_site_202503/binding_site_token_cls_50.csv",
    #     "/home/tanyang/workspace/VenusScope/data/token_cls_dataset/mix_family/binding_site/sim_50"
    # )
    stats = count_csv_rows('/home/tanyang/workspace/VenusScope/data/token_cls_dataset/mix_family/binding_site/sim_50')
    print("\nDetailed statistics:")
    for file_path, row_count in stats.items():
        print(f"{file_path}: {row_count} rows")
    
    stats = count_csv_rows('/home/tanyang/workspace/VenusScope/data/token_cls_dataset/mix_family/binding_site/sim_70')
    print("\nDetailed statistics:")
    for file_path, row_count in stats.items():
        print(f"{file_path}: {row_count} rows")

    stats = count_csv_rows('/home/tanyang/workspace/VenusScope/data/token_cls_dataset/mix_family/binding_site/sim_90')
    print("\nDetailed statistics:")
    for file_path, row_count in stats.items():
        print(f"{file_path}: {row_count} rows")
    

    # file_dir = '/home/tanyang/workspace/VenusScope/data/interpro_2503/'
    # process_interpro_labels(
    #     file_dir + 'domain/domain_token_cls_fragment_af2_unique_merged.csv'
    # )
    # process_interpro_labels(
    #     file_dir + 'domain/domain_token_cls_fragment_af2_unique_merged_50.csv'
    # )
    # process_interpro_labels(
    #     file_dir + 'domain/domain_token_cls_fragment_af2_unique_merged_70.csv'
    # )
    # process_interpro_labels(
    #     file_dir + 'domain/domain_token_cls_fragment_af2_unique_merged_90.csv'
    # )
    
    # file_dir = '/home/tanyang/workspace/VenusScope/data/'

    # add_interpro_labels_from_master(
    #     file_dir + 'token_cls_dataset/cross_family/domain',
    #     file_dir + 'interpro_2503/domain/domain_token_cls_fragment_af2_unique_merged.csv'
    # )
    
    # add_interpro_labels_from_master(
    #     file_dir + 'token_cls_dataset/cross_family/motif',
    #     file_dir + 'interpro_2503/motif/motif_token_cls_fragment_af2_unique_merged_90.csv'
    # )
    
    # add_interpro_labels_from_master(
    #     file_dir + 'token_cls_dataset/cross_family/active_site',
    #     file_dir + 'interpro_2503/active_site/active_site_token_cls_fragment_af2_unique_merged_70.csv'
    # )
    
    # add_interpro_labels_from_master(
    #     file_dir + 'token_cls_dataset/cross_family/binding_site',
    #     file_dir + 'interpro_2503/binding_site/binding_site_token_cls_fragment_af2_unique_merged_50.csv'
    # )

    # add_interpro_labels_from_master(
    #     file_dir + 'token_cls_dataset/cross_family/conserved_site',
    #     file_dir + 'interpro_2503/conserved_site/conserved_site_token_cls_fragment_af2_unique_merged_50.csv'
    # )

