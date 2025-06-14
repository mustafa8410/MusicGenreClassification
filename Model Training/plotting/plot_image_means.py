import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = [
    '../rounded_results/customcnn1_results_rounded.csv',
    '../rounded_results/customcnn2_results_rounded.csv',
    '../rounded_results/gtzan_cnn_corrected_results.csv',

]

all_dfs = []
for fp in file_paths:
    df = pd.read_csv(fp)
    # Standardize column names
    df = df.rename(columns={
        'Image Type': 'ImageType',
        'F1 Score': 'F1',
        "Cohen's Kappa": 'Cohen_Kappa',
        '5-Fold Accuracy': 'Accuracy'
    })
    if 'Model' not in df.columns:
        df['Model'] = 'N/A'
    # Select only relevant columns
    df = df[['ImageType', 'Accuracy', 'F1', 'Cohen_Kappa']]
    all_dfs.append(df)

# Combine all dataframes
full_df = pd.concat(all_dfs, ignore_index=True)

# Convert to numeric (for safety)
for col in ['Accuracy', 'F1', 'Cohen_Kappa']:
    full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

# Calculate mean metrics per image type
mean_metrics = full_df.groupby('ImageType')[['Accuracy', 'F1', 'Cohen_Kappa']].mean().reset_index()

image_type_order = ['melspectrogram', 'spectrogram', 'chromagram']
mean_metrics['ImageType'] = pd.Categorical(mean_metrics['ImageType'], categories=image_type_order, ordered=True)
mean_metrics = mean_metrics.sort_values('ImageType')

# Save to CSV
mean_metrics.to_csv('csv files/mean_metrics_per_image_type.csv', index=False, float_format="%.2f")
print("Mean metrics table saved as 'mean_metrics_per_image_type.csv'.")

# Grouped bar chart
metrics = ['Accuracy', 'F1', 'Cohen_Kappa']
bar_width = 0.25
x = range(len(mean_metrics))

plt.figure(figsize=(8, 6))
for i, metric in enumerate(metrics):
    bars = plt.bar(
        [pos + i*bar_width for pos in x],
        mean_metrics[metric],
        width=bar_width,
        label=metric.replace('_', ' ')
    )
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=8
        )  # <-- Added

plt.xticks([pos + bar_width for pos in x], mean_metrics['ImageType'])
plt.xlabel('Image Type')
plt.ylabel('Mean Value (%)')
plt.title("Mean Accuracy, F1 Score, and Cohen's Kappa by Image Type")
plt.legend()
plt.tight_layout()
plt.savefig("plots/mean_metrics_per_image_type.png")
print("Plot saved as 'mean_metrics_per_image_type.png'.")
plt.show()
