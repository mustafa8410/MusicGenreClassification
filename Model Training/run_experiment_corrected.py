import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
from sklearn.model_selection import KFold
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
    data_path = f'transformingAudio/{image_type}_images'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return datasets.ImageFolder(root=data_path, transform=transform)


def modify_model(name, num_classes):
    if name == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mobilenet_v2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)


def train_model(model, model_name, train_loader, val_loader, epochs=5, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"Training {model_name} for {epochs} epochs...")

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


def run_experiment(image_type, model_name, folds=5, batch_size=32, epochs=10):
    dataset = get_dataset(image_type)
    print("Dataset is loaded.")
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    num_classes = len(dataset.classes)
    scores = []
    print(f"\nImage Type: {image_type}, Model: {model_name}, Folds: {folds}, Batch Size: {batch_size}, Epochs: {epochs}")

    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*20} Fold {fold+1}/{folds} {'='*20}")

        train_val_data = Subset(dataset, train_val_idx)
        test_data = Subset(dataset, test_idx)

        train_size = int(0.8 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        train_data, val_data = random_split(train_val_data, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = modify_model(model_name, num_classes)
        print(f"Model {model_name} is loaded.")
        train_model(model, model_name, train_loader, val_loader, epochs)
        acc, prec, rec, f1, kappa = evaluate_model(model, test_loader)

        print(f"Fold {fold+1}: Accuracy={acc:.2f}%, Precision={prec:.2f}%, Recall={rec:.2f}%, F1={f1:.2f}%, Cohen's Kappa={kappa:.2f}%")
        scores.append((acc, prec, rec, f1, kappa))

    return np.mean(scores, axis=0)


# Main script
image_types = ['melspectrogram', 'spectrogram', 'chromagram']
models_to_test = ['resnet18', 'mobilenet_v2', 'efficientnet_b0']

results = []

for image_type in image_types:
    for model_name in models_to_test:
        avg_acc, avg_prec, avg_rec, avg_f1, avg_kappa = run_experiment(image_type, model_name)
        results.append({
            'Image Type': image_type,
            'Model': model_name,
            '5-Fold Accuracy': round(avg_acc, 2),
            'Precision': round(avg_prec, 2),
            'Recall': round(avg_rec, 2),
            'F1 Score': round(avg_f1, 2),
            "Cohen's Kappa": round(avg_kappa, 2)
        })
        print(f"\n{model_name} on {image_type} → Accuracy: {avg_acc:.2f}%, Precision: {avg_prec:.2f}%, Recall: {avg_rec:.2f}%, F1: {avg_f1:.2f}%, Cohen's Kappa: {avg_kappa:.2f}%\n")

# Save and print summary
df = pd.DataFrame(results)
df.to_csv("gtzan_cnn_corrected_results.csv", index=False)
print("\nFinal Results:\n")
print(df.to_string(index=False))
