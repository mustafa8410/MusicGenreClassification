import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
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


def train_model(model, model_name, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(f"Training {model_name} for {epochs} epochs...")

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
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

    print("Validation...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * np.mean(np.array(y_true) == np.array(y_pred))
    precision = 100 * precision_score(y_true, y_pred, average='macro')
    recall = 100 * recall_score(y_true, y_pred, average='macro')
    f1 = 100 * f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%")
    return accuracy, precision, recall, f1


def run_experiment(image_type, model_name, folds=5, batch_size=16, epochs=5):
    dataset = get_dataset(image_type)
    print("Dataset is loaded.")
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    num_classes = len(dataset.classes)
    scores = []
    print("\n\n\nStarting K-Fold Cross Validation for all models and data types...")
    print(f"\nImage Type: {image_type}, Model: {model_name}, Folds: {folds}, Batch Size: {batch_size}, Epochs: {epochs}")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'='*20} Fold {fold+1}/{folds} {'='*20}")
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, val_idx)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        model = modify_model(model_name, num_classes)
        print(f"Model {model_name} is loaded.")
        acc, prec, rec, f1 = train_model(model, model_name, train_loader, val_loader, epochs)
        scores.append((acc, prec, rec, f1))

    return np.mean(scores, axis=0)


# Main script
image_types = ['melspectrogram', 'spectrogram', 'chromagram']
models_to_test = ['resnet18', 'mobilenet_v2', 'efficientnet_b0']

results = []

for image_type in image_types:
    for model_name in models_to_test:
        avg_acc, avg_prec, avg_rec, avg_f1 = run_experiment(image_type, model_name)
        results.append({
            'Image Type': image_type,
            'Model': model_name,
            '5-Fold Accuracy': round(avg_acc, 2),
            'Precision': round(avg_prec, 2),
            'Recall': round(avg_rec, 2),
            'F1 Score': round(avg_f1, 2)
        })
        print(f"\n{model_name} on {image_type} → Accuracy: {avg_acc:.2f}%, Precision: {avg_prec:.2f}%, Recall: {avg_rec:.2f}%, F1: {avg_f1:.2f}%\n")

# Save and print summary
df = pd.DataFrame(results)
df.to_csv("gtzan_cnn_results.csv", index=False)
print("\nFinal Results:\n")
print(df.to_string(index=False))
