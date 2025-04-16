import argparse
import os
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge CSV files')
    parser.add_argument('--interpro_keyword_dir', type=str, help='Path to the interpro keyword directory')
    parser.add_argument('--csv_file_name', type=str, default='token_cls_fragment_af2_unique_merged.csv', help='Name of the CSV file to merge')
    parser.add_argument('--output_csv_file', type=str, required=True, help='Path to the output CSV file')
    args = parser.parse_args()
    
    interpro_ids = sorted(os.listdir(args.interpro_keyword_dir))
    print(f"Processing {len(interpro_ids)} interpro keywords")
    
    all_dfs = []
    
    for interpro_id in tqdm(interpro_ids):
        csv_path = os.path.join(args.interpro_keyword_dir, interpro_id, args.csv_file_name)
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} does not exist")
            continue
        
        try:
            # 检查文件是否为空
            if os.path.getsize(csv_path) == 0:
                print(f"CSV file {csv_path} is empty")
                continue
                
            df = pd.read_csv(csv_path)
            if not df.empty:
                all_dfs.append(df)
            else:
                print(f"CSV file {csv_path} has no data")
        except pd.errors.EmptyDataError:
            print(f"CSV file {csv_path} is empty or has no columns")
            continue
        except Exception as e:
            print(f"Error reading {csv_path}: {str(e)}")
            continue
    
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Merged {len(all_dfs)} CSV files with total {len(merged_df)} rows")
        
        merged_df.to_csv(args.output_csv_file, index=False)
        print(f"Saved merged CSV to {args.output_csv_file}")
    else:
        print("No CSV files were found to merge")
