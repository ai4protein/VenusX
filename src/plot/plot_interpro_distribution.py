import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

# set font size
plt.rcParams.update({'font.size': 16})

# define all site types
site_types = ['active_site', 'binding_site', 'conserved_site', 'domain', 'motif']

# define fragment length thresholds for each site type
fragment_thresholds = {
    'domain': 600,
    'motif': 230,
    'active_site': 1000,
    'binding_site': 1000,
    'conserved_site': 1000
}

# create a large figure with all subplots, adjust the aspect ratio
fig, axes = plt.subplots(2, len(site_types), figsize=(20, 8))

# create distribution plots for each site type
for idx, site_type in enumerate(site_types):
    # read the corresponding CSV file
    df = pd.read_csv(f'data/interpro_2503/{site_type}/{site_type}_token_cls_af2.csv')
    
    # calculate the length of full sequences and truncate
    df['seq_full_length'] = df['seq_full'].str.len()
    df = df[df['seq_full_length'] <= 3000]  # truncate full sequences to 3000
    
    # count the frequency of each InterPro type
    interpro_counts = df['interpro_label'].value_counts()
    
    # statistics
    full_min = interpro_counts.min()
    full_max = interpro_counts.max()
    full_mean = interpro_counts.mean()
    full_median = interpro_counts.median()
    
    # print statistics
    print(f"\n------- {site_type} -------")
    print(f"Number of unique InterPro types: {len(interpro_counts)}")
    
    # detailed statistics
    print(f"\nFull Sequence Statistics:")
    print(f"  Min: {full_min:.0f}")
    print(f"  Max: {full_max:.0f}")
    print(f"  Mean: {full_mean:.2f}")
    print(f"  Median: {full_median:.2f}")
    
    print("\nTop 5 most frequent InterPro types:")
    for label, count in interpro_counts.head(5).items():
        print(f"  {label}: {count}")
    
    # create a new DataFrame for fragment data
    fragment_data = []
    
    # process the fragment frequency for each interpro_label
    for interpro_label in df['interpro_label'].unique():
        # get the data for this label
        label_df = df[df['interpro_label'] == interpro_label]
        
        # calculate the number of fragments for this label
        fragment_count = 0
        for fragments in label_df['seq_fragment']:
            if pd.isna(fragments):  # handle possible empty values
                continue
            fragment_list = [frag.strip() for frag in fragments.split('|') if frag.strip()]
            # use different thresholds for different types
            threshold = fragment_thresholds[site_type]
            fragment_list = [frag for frag in fragment_list if len(frag) <= threshold]
            fragment_count += len(fragment_list)
        
        # add to the result list
        if fragment_count > 0:  # only record cases with fragments
            fragment_data.append({
                'interpro_label': interpro_label,
                'fragment_count': fragment_count
            })
    
    # create a DataFrame for fragment statistics
    fragment_df = pd.DataFrame(fragment_data)
    
    # if there is data, calculate statistics
    if not fragment_df.empty:
        fragment_counts = fragment_df['fragment_count']
        
        # statistics
        fragment_min = fragment_counts.min()
        fragment_max = fragment_counts.max()
        fragment_mean = fragment_counts.mean()
        fragment_median = fragment_counts.median()
        
        # print fragment statistics
        print(f"\nFragment Statistics:")
        print(f"  Number of InterPro types with fragments: {len(fragment_df)}")
        print(f"  Min: {fragment_min:.0f}")
        print(f"  Max: {fragment_max:.0f}")
        print(f"  Mean: {fragment_mean:.2f}")
        print(f"  Median: {fragment_median:.2f}")
        
        # print the top 5 most frequent fragment InterPro types
        print("\nTop 5 most frequent fragment InterPro types:")
        top_fragments = fragment_df.sort_values(by='fragment_count', ascending=False).head(5)
        for _, row in top_fragments.iterrows():
            print(f"  {row['interpro_label']}: {row['fragment_count']}")
        
        # plot the boxplot of fragment frequency (first row)
        # use logarithmic scale to handle the unbalanced distribution
        sns.boxplot(y=fragment_counts, ax=axes[0, idx], color='#3498db')
        axes[0, idx].set_title(f'{site_type.replace("_", " ").title()}', fontsize=22)
        axes[0, idx].set_yscale('log')  # use logarithmic scale
        
        # set y-axis ticks, ensure the minimum value is not less than 1
        min_val = max(1, fragment_counts.min())  # ensure the minimum value is not less than 1
        max_val = fragment_counts.max()
        
        # generate appropriate ticks
        ticks = []
        current = 10 ** np.floor(np.log10(min_val))
        while current <= max_val:
            ticks.append(current)
            current *= 10
        
        axes[0, idx].set_yticks(ticks)
        
        # only show y-axis label on the leftmost side
        if idx == 0:
            axes[0, idx].set_ylabel('Log Count', fontsize=20)
        else:
            axes[0, idx].set_ylabel('')
        axes[0, idx].tick_params(axis='y', labelsize=18)
    else:
        print(f"\n{site_type} has no fragment data")
        axes[0, idx].text(0.5, 0.5, 'No Data', horizontalalignment='center', 
                          verticalalignment='center', transform=axes[0, idx].transAxes)
    
    # plot the boxplot of full sequence frequency (second row)
    sns.boxplot(y=interpro_counts, ax=axes[1, idx], color='#2ecc71')
    axes[1, idx].set_yscale('log')  # use logarithmic scale
    
    # set y-axis ticks, ensure the minimum value is not less than 1
    min_val = max(1, interpro_counts.min())  # ensure the minimum value is not less than 1
    max_val = interpro_counts.max()
    
    # generate appropriate ticks
    ticks = []
    current = 10 ** np.floor(np.log10(min_val))
    while current <= max_val:
        ticks.append(current)
        current *= 10
    
    axes[1, idx].set_yticks(ticks)
    
    # only show y-axis label on the leftmost side
    if idx == 0:
        axes[1, idx].set_ylabel('Log Count', fontsize=20)
    else:
        axes[1, idx].set_ylabel('')
    axes[1, idx].tick_params(axis='y', labelsize=18)

# add legend
handles = [plt.Rectangle((0,0),1,1, color='#3498db'), plt.Rectangle((0,0),1,1, color='#2ecc71')]
labels = ['Fragment', 'Full']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
          ncol=2, fontsize=22)

# adjust the spacing between subplots
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9, hspace=0.3, wspace=0.2)

# save the figure
plt.savefig('interpro_distribution_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

