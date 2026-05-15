import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

S2P_COLOR = 'darkorange'

# Read the CSV
df = pd.read_csv('06_ec_group_stats.csv')

# Extract the data
ec_names = df['ec_name'].values
n_proteins = df['n_comparable_proteins'].values
n_s2p_pockets = df['n_s2p_pockets_total'].values
n_s2p_unique = df['n_s2p_unique_pockets'].values

# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Set the width of bars and positions
x = np.arange(len(ec_names))
width = 0.25

# Create bars on primary axis (S2P pockets)
bars2 = ax1.bar(x - width/2, n_s2p_pockets, width, label='Number of S2P Pockets', color='red', alpha=0.8)
bars3 = ax1.bar(x + width/2, n_s2p_unique, width, label='Number of Unique S2P Pockets', color=S2P_COLOR, alpha=0.8)

# Create second y-axis for proteins
ax2 = ax1.twinx()
bars1 = ax2.bar(x, n_proteins, width, label='Number of Proteins', alpha=0.6, color='#2ca02c')

# Customize the plot
ax1.set_xlabel('EC Enzyme Class', fontsize=12)
ax1.set_ylabel('Number of Pockets', fontsize=12)
ax2.set_ylabel('Number of Proteins', fontsize=12)
ax1.set_title('Proteins and S2P Pocket Statistics by EC Class', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(ec_names, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('ec_stats_plot.pdf', dpi=300, bbox_inches='tight')
print("Plot saved to: ec_stats_plot.pdf")
