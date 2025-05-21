import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


def get_dataset(image_type):
    data_path = f'transformingAudio/{image_type}_images'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return datasets.ImageFolder(root=data_path, transform=transform)

# 32 of 3x3 kernels, then normalized and relu
# 32 of 3x3 kernels, normalized and relu, then 2x2 pooling
# 64 of 3x3 kernels, norm, relu
# 64 of 3x3 kernels, norm, relu, 2x2 pooling
# 128 of 3x3 kernels, norm, relu
# 128 of 3x3 kernels, norm, relu, 2x2 pooling
# 256 of 3x3 kernels, norm, relu
# 256 of 3x3 kernels, norm, relu, 2x2 pooling
# flattened to vector
# fully connected layer, 256 input 256 output, relu, 0.5 dropout
# final classification layer

def build_sequential_cnn(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )


def train_and_evaluate(model, train_loader, val_loader, epochs=50):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = 100 * accuracy_score(y_true, y_pred)
    prec = 100 * precision_score(y_true, y_pred, average='weighted')
    rec = 100 * recall_score(y_true, y_pred, average='weighted')
    f1 = 100 * f1_score(y_true, y_pred, average='weighted')
    return acc, prec, rec, f1


def run_experiment(image_type, folds=5, batch_size=32, epochs=50):
    print(f"\n========= Training CustomCNN on {image_type.upper()} =========")
    dataset = get_dataset(image_type)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    num_classes = len(dataset.classes)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/{folds} ---")
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        model = build_sequential_cnn(num_classes)
        acc, prec, rec, f1 = train_and_evaluate(model, train_loader, val_loader, epochs)
        print(f"Fold {fold+1}: Accuracy={acc:.2f}%, Precision={prec:.2f}%, Recall={rec:.2f}%, F1={f1:.2f}%")
        scores.append((acc, prec, rec, f1))

    # Compute average scores
    scores_np = np.array(scores)
    mean_scores = scores_np.mean(axis=0)
    print(f"\nMean for {image_type.upper()}: Accuracy={mean_scores[0]:.2f}%, Precision={mean_scores[1]:.2f}%, Recall={mean_scores[2]:.2f}%, F1={mean_scores[3]:.2f}%")
    return (image_type, *mean_scores)


# Run for all image types
all_results = []
for image_type in ["spectrogram", "melspectrogram", "chromagram"]:
    all_results.append(run_experiment(image_type))

results_df = pd.DataFrame(all_results, columns=["ImageType", "Accuracy", "Precision", "Recall", "F1"])
print("\n===== AVERAGE RESULTS FOR CUSTOM CNN =====")
print(results_df)
results_df.to_csv("customcnn1_results.csv", index=False)
print("Results saved to 'customcnn1_results.csv'.")
