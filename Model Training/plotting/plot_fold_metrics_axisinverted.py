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

# 1. Grouped bar chart for Cohen's Kappa (x: models, bars: image types)
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(10, 6))
bar_width = 0.22
x = range(len(model_order))

for i, img_type in enumerate(image_type_order):
    values = []
    for model in model_order:
        val = df[(df['ImageType'] == img_type) & (df['Model'] == model)]['Cohen_Kappa']
        values.append(val.values[0] if not val.empty else 0)
    bars = plt.bar(
        [pos + i*bar_width for pos in x],
        values,
        width=bar_width,
        label=img_type
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
        )

plt.xticks([pos + bar_width for pos in x], model_order)
plt.xlabel('Model')
plt.ylabel("Cohen's Kappa (%)")
plt.title("Cohen's Kappa by Image Type and Model (Best Folds)")
plt.legend(title='Image Type')
plt.tight_layout()
plt.savefig('plots/best_folds_cohen_kappa_xmodel.png')
plt.show()

# 2. Grouped bar chart for F1 Score (x: models, bars: image types)
plt.figure(figsize=(10, 6))
for i, img_type in enumerate(image_type_order):
    values = []
    for model in model_order:
        val = df[(df['ImageType'] == img_type) & (df['Model'] == model)]['F1']
        values.append(val.values[0] if not val.empty else 0)
    bars = plt.bar(
        [pos + i*bar_width for pos in x],
        values,
        width=bar_width,
        label=img_type
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
        )

plt.xticks([pos + bar_width for pos in x], model_order)
plt.xlabel('Model')
plt.ylabel("F1 Score (%)")
plt.title("F1 Score by Image Type and Model (Best Folds)")
plt.legend(title='Image Type')
plt.tight_layout()
plt.savefig('plots/best_folds_f1_score_xmodel.png')
plt.show()
