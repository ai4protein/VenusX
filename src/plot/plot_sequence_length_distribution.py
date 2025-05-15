import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

# set the global font size
plt.rcParams.update({'font.size': 16})

# define all site types
site_types = ['active_site', 'binding_site', 'conserved_site', 'domain', 'motif']

# define the fragment length thresholds for each type
fragment_thresholds = {
    'domain': 600,
    'motif': 230,
    'active_site': 1000,
    'binding_site': 1000,
    'conserved_site': 1000
}

# create a large figure with all subplots, adjust the aspect ratio
fig, axes = plt.subplots(2, len(site_types), figsize=(20, 8))

# create a distribution plot for each site type
for idx, site_type in enumerate(site_types):
    # read the corresponding CSV file
    df = pd.read_csv(f'data/interpro_2503/{site_type}/{site_type}_token_cls_af2.csv')
    
    # calculate the length of full sequences and truncate
    df['seq_full_length'] = df['seq_full'].str.len()
    df = df[df['seq_full_length'] <= 3000]  # truncate full sequences in 3000
    
    # process the length of fragment sequences (may have multiple fragments)
    fragment_lengths = []
    for fragments in df['seq_fragment']:
        # split multiple fragments and calculate the length of each fragment
        if pd.isna(fragments):  # handle possible empty values
            continue
        lengths = [len(frag.strip()) for frag in fragments.split('|') if frag.strip()]  # remove white spaces
        # use different thresholds for different types
        threshold = fragment_thresholds[site_type]
        lengths = [length for length in lengths if length <= threshold]
        fragment_lengths.extend(lengths)
    
    # calculate statistics
    fragment_mean = np.mean(fragment_lengths) if fragment_lengths else 0
    fragment_median = np.median(fragment_lengths) if fragment_lengths else 0
    full_mean = df['seq_full_length'].mean()
    full_median = df['seq_full_length'].median()
    
    # print some statistics to check the data
    print(f"\n------- {site_type} -------")
    print(f"Fragment lengths:")
    print(f"  Number of fragments: {len(fragment_lengths)}")
    print(f"  Min length: {min(fragment_lengths) if fragment_lengths else 0}")
    print(f"  Max length: {max(fragment_lengths) if fragment_lengths else 0}")
    print(f"  Mean length: {fragment_mean:.2f}")
    print(f"  Median length: {fragment_median:.2f}")
    
    print(f"\nFull sequence lengths:")
    print(f"  Number of sequences: {len(df)}")
    print(f"  Min length: {df['seq_full_length'].min()}")
    print(f"  Max length: {df['seq_full_length'].max()}")
    print(f"  Mean length: {full_mean:.2f}")
    print(f"  Median length: {full_median:.2f}")
    
    # plot the distribution of fragment sequence lengths (first row)
    sns.histplot(data=fragment_lengths, bins=30, ax=axes[0, idx], color='#3498db', label='Fragment')
    axes[0, idx].set_title(f'{site_type.replace("_", " ").title()}', fontsize=22)
    axes[0, idx].set_xlabel('Length', fontsize=20)
    
    # add the mean and median vertical lines
    if fragment_lengths:
        axes[0, idx].axvline(x=fragment_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {fragment_mean:.1f}')
        axes[0, idx].axvline(x=fragment_median, color='green', linestyle='--', linewidth=2, label=f'Median: {fragment_median:.1f}')
        # add text labels in the plot
        y_max = axes[0, idx].get_ylim()[1]
        axes[0, idx].text(fragment_mean + 5, y_max * 0.95, f'Mean: {fragment_mean:.1f}', color='red', fontsize=14)
        axes[0, idx].text(fragment_median + 5, y_max * 0.85, f'Median: {fragment_median:.1f}', color='green', fontsize=14)
    
    # only show y-axis label on the leftmost side
    if idx == 0:
        axes[0, idx].set_ylabel('Count (k)', fontsize=20)
    else:
        axes[0, idx].set_ylabel('')
    axes[0, idx].tick_params(axis='both', which='major', labelsize=18)
    # set y-axis ticks to k units, decide whether to show decimals based on whether the value is an integer
    axes[0, idx].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: '' if x == 0 else (f'{x/1000:.1f}' if x/1000 != int(x/1000) else f'{x/1000:.0f}')
    ))
    
    # plot the distribution of full sequence lengths (second row)
    sns.histplot(data=df, x='seq_full_length', bins=30, ax=axes[1, idx], color='#2ecc71', label='Full')
    axes[1, idx].set_xlabel('Length', fontsize=20)
    
    # add the mean and median vertical lines
    axes[1, idx].axvline(x=full_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {full_mean:.1f}')
    axes[1, idx].axvline(x=full_median, color='green', linestyle='--', linewidth=2, label=f'Median: {full_median:.1f}')
    # add text labels in the plot
    y_max = axes[1, idx].get_ylim()[1]
    axes[1, idx].text(full_mean + 50, y_max * 0.95, f'Mean: {full_mean:.1f}', color='red', fontsize=14)
    axes[1, idx].text(full_median + 50, y_max * 0.85, f'Median: {full_median:.1f}', color='green', fontsize=14)
    
    # only show y-axis label on the leftmost side
    if idx == 0:
        axes[1, idx].set_ylabel('Count (k)', fontsize=20)
    else:
        axes[1, idx].set_ylabel('')
    axes[1, idx].tick_params(axis='both', which='major', labelsize=18)
    # set y-axis ticks to k units, decide whether to show decimals based on whether the value is an integer
    axes[1, idx].yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: '' if x == 0 else (f'{x/1000:.1f}' if x/1000 != int(x/1000) else f'{x/1000:.0f}')
    ))

# add legend
# get the handles and labels of the first subplot
handles1, labels1 = axes[0, 0].get_legend_handles_labels()
# get the handles and labels of the second subplot
handles2, labels2 = axes[1, 0].get_legend_handles_labels()
# merge handles and labels
handles = handles1 + handles2
labels = labels1 + labels2
# create legend
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
          ncol=4, fontsize=22)

# adjust the spacing between subplots
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9, hspace=0.3, wspace=0.2)

# save the figure
plt.savefig('sequence_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
