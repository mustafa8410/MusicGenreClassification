import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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
    ])
    return datasets.ImageFolder(root=data_path, transform=transform)

def modify_model(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

def train_model(model, train_loader, val_loader, epochs=50, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Current LR: {scheduler.get_last_lr()} for epoch {epoch + 1}")
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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
    prec = 100 * precision_score(y_true, y_pred, average='macro')
    rec = 100 * recall_score(y_true, y_pred, average='macro')
    f1 = 100 * f1_score(y_true, y_pred, average='macro')
    kappa = 100 * cohen_kappa_score(y_true, y_pred)
    return acc, prec, rec, f1, kappa

# ==== Main Experiment ====
image_type = 'melspectrogram'
model_name = 'efficientnet_b0'
folds = 5
batch_size = 32
epochs = 50

dataset = get_dataset(image_type)
labels = [label for _, label in dataset.imgs]
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
num_classes = len(dataset.classes)

fold_metrics = []
best_metric = -np.inf
best_fold = -1
best_model_weights = None

for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n{'='*20} Fold {fold+1}/{folds} {'='*20}")
    train_val_data = Subset(dataset, train_val_idx)
    test_data = Subset(dataset, test_idx)
    train_size = int(0.8 * len(train_val_data))
    val_size = len(train_val_data) - train_size
    train_data, val_data = random_split(train_val_data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = modify_model(num_classes)
    train_model(model, train_loader, val_loader, epochs)
    acc, prec, rec, f1, kappa = evaluate_model(model, test_loader)

    print(f"Fold {fold+1}: Accuracy={acc:.2f}%, Precision={prec:.2f}%, Recall={rec:.2f}%, F1={f1:.2f}%, Cohen's Kappa={kappa:.2f}%")
    fold_metrics.append({'Fold': fold+1, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, "Cohen's Kappa": kappa})


    if kappa > best_metric:
        best_metric = kappa
        best_fold = fold + 1
        best_model_weights = model.state_dict()

# Save all folds' metrics to CSV
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv("efficientnet_b0_melspectrogram_fold_metrics.csv", index=False)
print("\nAll fold metrics saved to 'efficientnet_b0_melspectrogram_fold_metrics.csv'.")

# Save the best model weights to disk
if best_model_weights is not None:
    model_save_path = f"efficientnet_b0_melspec.pth"
    torch.save(best_model_weights, model_save_path)
    print(f"Best model weights (fold {best_fold}) saved to '{model_save_path}'.")
