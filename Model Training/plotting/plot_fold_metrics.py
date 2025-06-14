import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'csv files/best_folds_allcnns_allimages.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Standardize columns
col_map = {
    'Image Type': 'ImageType',
    'F1 Score': 'F1',
    "Cohen's Kappa": 'Cohen_Kappa',
}
df = df.rename(columns=col_map)

# Define order for image types and models
image_type_order = ['melspectrogram', 'spectrogram', 'chromagram']
model_order = ['resnet18', 'mobilenet_v2', 'efficientnet_b0', 'CustomCNN-1', 'CustomCNN-2']

df['ImageType'] = pd.Categorical(df['ImageType'], categories=image_type_order, ordered=True)
df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
df = df.sort_values(['Model', 'ImageType'])

# 1. Grouped bar chart for Cohen's Kappa
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(10, 6))
bar_width = 0.15
x = range(len(image_type_order))

for i, model in enumerate(model_order):
    values = []
    for img_type in image_type_order:
        val = df[(df['ImageType'] == img_type) & (df['Model'] == model)]['Cohen_Kappa']
        values.append(val.values[0] if not val.empty else 0)
    bars = plt.bar(
        [pos + i*bar_width for pos in x],
        values,
        width=bar_width,
        label=model
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

plt.xticks([pos + (len(model_order)/2 - 0.5)*bar_width for pos in x], image_type_order)
plt.xlabel('Image Type')
plt.ylabel("Cohen's Kappa (%)")
plt.title("Cohen's Kappa by Model and Image Type (Best Folds)")
plt.legend()
plt.tight_layout()
plt.savefig('plots/best_folds_cohen_kappa.png')
plt.show()

# 2. Grouped bar chart for F1 Score
plt.figure(figsize=(10, 6))
for i, model in enumerate(model_order):
    values = []
    for img_type in image_type_order:
        val = df[(df['ImageType'] == img_type) & (df['Model'] == model)]['F1']
        values.append(val.values[0] if not val.empty else 0)
    bars = plt.bar(
        [pos + i*bar_width for pos in x],
        values,
        width=bar_width,
        label=model
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

plt.xticks([pos + (len(model_order)/2 - 0.5)*bar_width for pos in x], image_type_order)
plt.xlabel('Image Type')
plt.ylabel("F1 Score (%)")
plt.title("F1 Score by Model and Image Type (Best Folds)")
plt.legend()
plt.tight_layout()
plt.savefig('plots/best_folds_f1_score.png')
plt.show()
