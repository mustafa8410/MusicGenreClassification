import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'csv files/melspectrogram_results.csv'
df = pd.read_csv(file_path)

col_map = {
    'Image Type': 'ImageType',
    "Cohen's Kappa": 'Cohen_Kappa'
}
df = df.rename(columns=col_map)
df = df[df['ImageType'] == 'melspectrogram']

model_order = ['resnet18', 'mobilenet_v2', 'efficientnet_b0', 'CustomCNN-1', 'CustomCNN-2']
df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
df = df.sort_values('Model')

plt.figure(figsize=(8, 5))
bars = plt.bar(df['Model'], df['Cohen_Kappa'], color='steelblue')

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

plt.xlabel('Model')
plt.ylabel("Cohen's Kappa (%)")
plt.title("Cohen's Kappa of Models on Melspectrogram")
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/melspectrogram_cohen_kappa.png')
plt.show()
