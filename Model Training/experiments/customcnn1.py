import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


def get_dataset(image_type):
    data_path = f'../transformingAudio/{image_type}_images'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean= [0.5] * 3, std= [0.5] * 3)
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
        nn.Sigmoid(),
        nn.Dropout(0.15),
        nn.Linear(256, 128),
        nn.Sigmoid(),
        nn.Dropout(0.15),
        nn.Linear(128, num_classes)
    )


def train_model(model, train_loader, val_loader, epochs=75, patience=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

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

        avg_train_loss = running_loss / len(train_loader)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                loss = criterion(val_outputs, y_val)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Current LR: {scheduler.get_last_lr()} for epoch {epoch+1}")

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = 100 * accuracy_score(y_true, y_pred)
    prec = 100 * precision_score(y_true, y_pred, average='weighted')
    rec = 100 * recall_score(y_true, y_pred, average='weighted')
    f1 = 100 * f1_score(y_true, y_pred, average='weighted')
    kappa = 100 * cohen_kappa_score(y_true, y_pred)
    return acc, prec, rec, f1, kappa


# Create a single DataFrame to store fold results for all image types
all_fold_results_df = pd.DataFrame(columns=["ImageType", "Fold", "Accuracy", "Precision", "Recall", "F1", "Cohen's Kappa"])

def run_experiment(image_type, folds=5, batch_size=32, epochs=75):
    global all_fold_results_df  # To modify the global DataFrame
    print(f"\n========= Training CustomCNN on {image_type.upper()} =========")
    dataset = get_dataset(image_type)
    labels = [label for _, label in dataset.imgs]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    num_classes = len(dataset.classes)
    scores = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold+1}/{folds} ---")

        train_val_data = Subset(dataset, train_val_idx)
        test_data = Subset(dataset, test_idx)

        train_size = int(0.8 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        train_data, val_data = random_split(train_val_data, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = build_sequential_cnn(num_classes)
        train_model(model, train_loader, val_loader, epochs)
        acc, prec, rec, f1, kappa = evaluate_model(model, test_loader)

        print(f"Fold {fold+1}: Accuracy={acc:.2f}%, Precision={prec:.2f}%, Recall={rec:.2f}%, F1={f1:.2f}%, Cohen's Kappa={kappa:.2f}%")
        scores.append((acc, prec, rec, f1, kappa))

        # Append fold results to the global DataFrame
        all_fold_results_df.loc[len(all_fold_results_df)] = [image_type, fold + 1, acc, prec, rec, f1, kappa]

    scores_np = np.array(scores)
    mean_scores = scores_np.mean(axis=0)
    print(f"\nMean for {image_type.upper()}: Accuracy={mean_scores[0]:.2f}%, Precision={mean_scores[1]:.2f}%, Recall={mean_scores[2]:.2f}%, F1={mean_scores[3]:.2f}%, Cohen's Kappa={mean_scores[4]:.2f}%")
    return (image_type, *mean_scores)


all_results = []
for image_type in ["spectrogram", "melspectrogram", "chromagram"]:
    all_results.append(run_experiment(image_type))

# Save mean scores
results_df = pd.DataFrame(all_results, columns=["ImageType", "Accuracy", "Precision", "Recall", "F1", "Cohen's Kappa"])
print("\n===== AVERAGE RESULTS FOR CUSTOM CNN =====")
print(results_df)
results_df.to_csv("customcnn1_corrected_results.csv", index=False)
print("Mean results saved to 'customcnn1_corrected_results.csv'.")

# Save all fold-wise results
all_fold_results_df.to_csv("customcnn1_all_fold_results.csv", index=False)
print("All fold-wise results saved to 'customcnn1_all_fold_results.csv'.")

