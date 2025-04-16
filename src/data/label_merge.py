import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def merge_labels(df):
    """
    Merge rows with the same uid and seq_full,
    and merge their labels (take the maximum value, equivalent to the union operation)
    """
    # group by uid and seq_full
    grouped = df.groupby(['uid', 'seq_full'])
    
    merged_records = []
    for (uid, seq_full), group in grouped:
        # if there is only one row, add it directly
        if len(group) == 1:
            merged_records.append(group.iloc[0].to_dict())
            continue
            
        # if there are multiple rows, merge labels
        labels = group['label'].tolist()
        merged_label = np.maximum.reduce([np.array(label) for label in labels])
        
        # get all fragments and their positions
        fragments = []
        positions = []
        for _, row in group.iterrows():
            fragments.append(row['seq_fragment'])
            positions.append(f"{row['start']}-{row['end']}")
        
        # create merged record
        merged_record = {
            'uid': uid,
            'interpro_id': group['interpro_id'].iloc[0],
            'seq_full': seq_full,
            'seq_fragment': '|'.join(fragments),
            'start': '|'.join([str(x) for x in group['start']]),
            'end': '|'.join([str(x) for x in group['end']]),
            'label': merged_label.tolist()
        }
        merged_records.append(merged_record)
    
    return pd.DataFrame(merged_records)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge labels for same sequences')
    parser.add_argument('--interpro_keyword_dir', type=str, required=True, help='Path to the interpro keyword directory')
    args = parser.parse_args()
    
    interpro_ids = sorted(os.listdir(os.path.join(args.interpro_keyword_dir, 'raw')))
    print(f"Processing {len(interpro_ids)} interpro keywords")
    
    for interpro_id in tqdm(interpro_ids):
        csv_path = os.path.join(args.interpro_keyword_dir, 'raw', interpro_id, 'token_cls_fragment_af2_unique.csv')
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} does not exist")
            continue

        df = pd.read_csv(csv_path)
        
        df['label'] = df['label'].apply(eval)
        
        # merge labels within a sequence
        merged_df = merge_labels(df)
        if len(merged_df) != len(df):
            print(f"{interpro_id}: Merged {len(df)} rows to {len(merged_df)} rows")
        # save merged results
        output_path = os.path.join(args.interpro_keyword_dir, 'raw', interpro_id, 'token_cls_fragment_af2_unique_merged.csv')
        merged_df.to_csv(output_path, index=False)
