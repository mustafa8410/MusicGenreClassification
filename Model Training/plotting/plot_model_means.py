import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'csv files/all_results.csv'
df = pd.read_csv(file_path)

# Standardize columns
col_map = {
    'Image Type': 'ImageType',
    '5-Fold Accuracy': 'Accuracy',
    'F1 Score': 'F1',
    "Cohen's Kappa": 'Cohen_Kappa'
}
df = df.rename(columns=col_map)

# Compute mean metrics per model
mean_per_model = df.groupby('Model')[['Accuracy', 'F1', 'Cohen_Kappa']].mean().reset_index()

# Optional: specify model order
model_order = ['resnet18', 'mobilenet_v2', 'efficientnet_b0', 'CustomCNN-1', 'CustomCNN-2']
mean_per_model['Model'] = pd.Categorical(mean_per_model['Model'], categories=model_order, ordered=True)
mean_per_model = mean_per_model.sort_values('Model')

# Save to CSV with two decimals
os.makedirs('csv files', exist_ok=True)
mean_per_model.to_csv('csv files/mean_metrics_per_model.csv', index=False, float_format="%.2f")

# Grouped bar chart
metrics = ['Accuracy', 'F1', 'Cohen_Kappa']
bar_width = 0.25
x = range(len(mean_per_model))

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    bars = plt.bar(
        [pos + i*bar_width for pos in x],
        mean_per_model[metric],
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

plt.xticks([pos + bar_width for pos in x], mean_per_model['Model'])
plt.xlabel('Model')
plt.ylabel('Mean Value (%)')
plt.title("Mean Accuracy, F1 Score, and Cohen's Kappa by Model")
plt.legend()
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/mean_metrics_per_model.png")
plt.show()
