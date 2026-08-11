import os
import copy
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import timm

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")

# =====================================================
# CONFIGURATION
# =====================================================

SEED = 24

IMAGE_SIZE = 300

BATCH_SIZE = 16

NUM_CLASSES = 3

TL_EPOCHS = 24

LR = 1e-3

WEIGHT_DECAY = 1e-4

MODEL_NAME = "tf_efficientnetv2_b0"

DEVICE_NAMES = [
    "Macro",
    "Micro",
    "Phone"
]

# =====================================================
# DATASET PATHS
# =====================================================

BASE_DIR = r"/mnt/d/Projects/Mosquito Analysis"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "device_dataset",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "device_dataset",
    "val"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "device_dataset",
    "test"
)

# =====================================================
# RESULTS DIRECTORY
# =====================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "domain_TL"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

MISCLASSIFIED_DIR = os.path.join(
    RESULTS_DIR,
    "misclassified_images"
)

os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# =====================================================
# RANDOM SEED
# =====================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("="*70)
print("DEVICE :", DEVICE)
print("="*70)

# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(18),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# =====================================================
# DATASETS
# =====================================================

train_dataset = ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = ImageFolder(
    VAL_DIR,
    transform=val_transform
)

test_dataset = ImageFolder(
    TEST_DIR,
    transform=val_transform
)

# =====================================================
# CLASS INFORMATION
# =====================================================

print("\nClass Mapping")

print(train_dataset.class_to_idx)

CLASS_MAPPING = train_dataset.class_to_idx

with open(
    os.path.join(
        RESULTS_DIR,
        "class_mapping.json"
    ),
    "w"
) as f:

    json.dump(
        CLASS_MAPPING,
        f,
        indent=4
    )

print("\nDataset Statistics")

print("-"*60)

print("Training Images  :", len(train_dataset))

print("Validation Images:", len(val_dataset))

print("Testing Images   :", len(test_dataset))

print("-"*60)

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\nDataLoaders Created Successfully")

print("Train Batches :", len(train_loader))

print("Val Batches   :", len(val_loader))

print("Test Batches  :", len(test_loader))

# =====================================================
# CSV LOG FILE
# =====================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "training_log.csv"
)

with open(csv_path, "w") as f:

    f.write(
        "Epoch,"
        "TrainLoss,"
        "TrainAccuracy,"
        "ValidationLoss,"
        "ValidationAccuracy\n"
    )

print("\nInitialization Completed Successfully.")
print("="*70)

# =====================================================
# MODEL
# =====================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=0.3
)

print("\nImageNet Pretrained Model Loaded Successfully")

# =====================================================
# FREEZE BACKBONE
# =====================================================

for param in model.parameters():

    param.requires_grad = False

# Train ONLY classifier

for param in model.classifier.parameters():

    param.requires_grad = True

model = model.to(DEVICE)

# =====================================================
# PARAMETERS
# =====================================================

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print("\nModel Summary")
print("-"*60)
print(f"Trainable Parameters : {trainable:,}")
print(f"Total Parameters     : {total:,}")
print(f"Trainable Percentage : {(100*trainable/total):.2f}%")
print("-"*60)

# =====================================================
# LOSS FUNCTION
# =====================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# =====================================================
# HISTORY
# =====================================================

train_loss_history = []

val_loss_history = []

train_acc_history = []

val_acc_history = []

# =====================================================
# BEST WEIGHTS TRACKING (No Early Stopping Counter)
# =====================================================

best_val_loss = float("inf")

best_val_acc = 0.0

best_weights = copy.deepcopy(
    model.state_dict()
)

# =====================================================
# TRAINING STARTS
# =====================================================

print("\nStarting Domain Transfer Learning...\n")

for epoch in range(TL_EPOCHS):

    ####################################################
    # TRAIN
    ####################################################

    model.train()

    running_loss = 0.0

    running_correct = 0

    total_samples = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)

        running_correct += (preds == labels).sum().item()

        total_samples += labels.size(0)

    train_loss = running_loss / total_samples

    train_acc = running_correct / total_samples

    ####################################################
    # VALIDATION
    ####################################################

    model.eval()

    val_running_loss = 0.0

    val_running_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            val_running_loss += loss.item() * images.size(0)

            val_running_correct += (preds == labels).sum().item()

            val_total += labels.size(0)

    val_loss = val_running_loss / val_total

    val_acc = val_running_correct / val_total

    ####################################################
    # Save History
    ####################################################

    train_loss_history.append(train_loss)

    val_loss_history.append(val_loss)

    train_acc_history.append(train_acc)

    val_acc_history.append(val_acc)

    ####################################################
    # CSV
    ####################################################

    with open(csv_path, "a") as f:

        f.write(
            f"{epoch+1},"
            f"{train_loss:.6f},"
            f"{train_acc:.6f},"
            f"{val_loss:.6f},"
            f"{val_acc:.6f}\n"
        )

    ####################################################
    # Console
    ####################################################

    print(
        f"Epoch [{epoch+1:03d}/{TL_EPOCHS}] "
        f"Train Loss={train_loss:.4f} "
        f"Train Acc={train_acc:.4f} "
        f"Val Loss={val_loss:.4f} "
        f"Val Acc={val_acc:.4f}"
    )

    ####################################################
    # SAVE BEST MODEL
    ####################################################

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_val_acc = val_acc

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_weights,
            os.path.join(
                RESULTS_DIR,
                "best_domain_model.pth"
            )
        )

        print("Best Model Saved.")

# =====================================================
# SAVE LAST MODEL
# =====================================================

torch.save(
    model.state_dict(),
    os.path.join(
        RESULTS_DIR,
        "last_domain_model.pth"
    )
)

print("\nTraining Completed Successfully.")

print(f"Best Validation Accuracy : {best_val_acc:.4f}")

print(f"Best Validation Loss     : {best_val_loss:.4f}")

# =====================================================
# LOAD BEST MODEL
# =====================================================

print("\nLoading Best Model...")

best_model_path = os.path.join(
    RESULTS_DIR,
    "best_domain_model.pth"
)

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

# =====================================================
# TESTING
# =====================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []

test_loss = 0.0

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

with torch.no_grad():

    sample_index = 0

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        test_loss += loss.item() * images.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())

        # Image paths
        batch_size = images.size(0)

        for i in range(batch_size):

            img_path = test_dataset.samples[sample_index][0]

            all_paths.append(img_path)

            sample_index += 1

test_loss /= len(test_dataset)

print("\nTesting Completed.")

# =====================================================
# METRICS
# =====================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print("="*60)
print("TEST RESULTS")
print("="*60)

print(f"Loss      : {test_loss:.4f}")
print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =====================================================
# SAVE TEST RESULTS
# =====================================================

results_path = os.path.join(
    RESULTS_DIR,
    "test_results.txt"
)

with open(results_path, "w") as f:

    f.write("DOMAIN TRANSFER LEARNING RESULTS\n")
    f.write("="*60 + "\n")

    f.write(f"Test Loss      : {test_loss:.6f}\n")
    f.write(f"Accuracy       : {test_accuracy:.6f}\n")
    f.write(f"Precision      : {precision:.6f}\n")
    f.write(f"Recall         : {recall:.6f}\n")
    f.write(f"F1 Score       : {f1:.6f}\n")

print("Test Results Saved.")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=test_dataset.classes,
    digits=4
)

print(report)

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(report_path, "w") as f:

    f.write(report)

print("Classification Report Saved.")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax,
    colorbar=False
)

plt.title("Domain TL Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("Confusion Matrix Saved.")

# =====================================================
# SAVE PREDICTIONS CSV
# =====================================================

prediction_rows = []

for i in range(len(all_labels)):

    probs = all_probabilities[i]

    prediction_rows.append({
        "Image": os.path.basename(all_paths[i]),
        "Image_Path": all_paths[i],
        "True_Label": test_dataset.classes[all_labels[i]],
        "Predicted_Label": test_dataset.classes[all_predictions[i]],
        "Confidence": float(np.max(probs))
    })

prediction_df = pd.DataFrame(prediction_rows)

prediction_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "predictions.csv"
    ),
    index=False
)

print("Predictions CSV Saved.")

# =====================================================
# TRAINING SUMMARY
# =====================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "training_summary.txt"
)

with open(summary_path, "w") as f:

    f.write("="*70 + "\n")
    f.write("DOMAIN TRANSFER LEARNING SUMMARY\n")
    f.write("="*70 + "\n\n")

    f.write(f"Model               : {MODEL_NAME}\n")
    f.write(f"Image Size          : {IMAGE_SIZE}\n")
    f.write(f"Batch Size          : {BATCH_SIZE}\n")
    f.write(f"Epochs              : {TL_EPOCHS}\n")
    f.write(f"Learning Rate       : {LR}\n")
    f.write(f"Weight Decay        : {WEIGHT_DECAY}\n\n")

    f.write(f"Train Images : {len(train_dataset)}\n")
    f.write(f"Val Images   : {len(val_dataset)}\n")
    f.write(f"Test Images  : {len(test_dataset)}\n\n")

    f.write("FINAL TEST RESULTS\n")
    f.write("-"*50 + "\n")

    f.write(f"Loss      : {test_loss:.6f}\n")
    f.write(f"Accuracy  : {test_accuracy:.6f}\n")
    f.write(f"Precision : {precision:.6f}\n")
    f.write(f"Recall    : {recall:.6f}\n")
    f.write(f"F1 Score  : {f1:.6f}\n")

print("Training Summary Saved.")

# =====================================================
# LOSS CURVE
# =====================================================

plt.figure(figsize=(8,6))

plt.plot(
    train_loss_history,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    val_loss_history,
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    ),
    dpi=300
)

plt.close()

print("Loss Curve Saved.")

# =====================================================
# ACCURACY CURVE
# =====================================================

plt.figure(figsize=(8,6))

plt.plot(
    train_acc_history,
    label="Train Accuracy",
    linewidth=2
)

plt.plot(
    val_acc_history,
    label="Validation Accuracy",
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Accuracy Curve")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    ),
    dpi=300
)

plt.close()

print("Accuracy Curve Saved.")

# =====================================================
# SAVE FEATURE EXTRACTOR
# (Needed later for Fusion TL)
# =====================================================

encoder_state = {}

for key, value in model.state_dict().items():

    if not key.startswith("classifier"):

        encoder_state[key] = value

torch.save(
    encoder_state,
    os.path.join(
        RESULTS_DIR,
        "best_domain_encoder.pth"
    )
)

print("Domain Encoder Saved.")

# =====================================================
# SAVE MISCLASSIFIED IMAGES
# =====================================================

print("\nSaving Misclassified Images...")

for img_path, gt, pred in zip(
    all_paths,
    all_labels,
    all_predictions
):

    if gt == pred:

        continue

    true_name = test_dataset.classes[gt]

    pred_name = test_dataset.classes[pred]

    save_dir = os.path.join(
        MISCLASSIFIED_DIR,
        f"{true_name}_as_{pred_name}"
    )

    os.makedirs(save_dir, exist_ok=True)

    image = Image.open(img_path).convert("RGB")

    image.save(
        os.path.join(
            save_dir,
            os.path.basename(img_path)
        )
    )

print("Misclassified Images Saved.")

# =====================================================
# FINISHED
# =====================================================

print("\n" + "="*70)
print("DOMAIN TRANSFER LEARNING COMPLETED SUCCESSFULLY")
print("="*70)

print(f"Results saved in:\n{RESULTS_DIR}")
