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

# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 24
IMAGE_SIZE = 300
BATCH_SIZE = 16
NUM_CLASSES = 3

# UPDATED METHOD LOGIC
NUM_EPOCHS = 100      
PATIENCE = 15        

LR_BACKBONE = 1e-4   
LR_HEAD = 1e-3      

WEIGHT_DECAY = 1e-4

MODEL_NAME = "tf_efficientnetv2_b0"

DEVICE_NAMES = [
    "Macro",
    "Micro",
    "Phone"
]

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = r"/mnt/d/Projects/Mosquito Analysis"

TRAIN_DIR = os.path.join(BASE_DIR, "device_dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "device_dataset", "val")
TEST_DIR = os.path.join(BASE_DIR, "device_dataset", "test")

TL_MODEL_PATH = os.path.join(BASE_DIR, "domain_TL", "best_domain_model.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "domain_FT")

os.makedirs(RESULTS_DIR, exist_ok=True)

MISCLASSIFIED_DIR = os.path.join(RESULTS_DIR, "misclassified_images")
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)

# ==========================================================
# RANDOM SEED
# ==========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print("DEVICE :", DEVICE)
print("="*70)

# ==========================================================
# TRANSFORMS
# ==========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(18),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================================
# DATASETS
# ==========================================================

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = ImageFolder(VAL_DIR, transform=test_transform)
test_dataset = ImageFolder(TEST_DIR, transform=test_transform)

print("\nClass Mapping")
print(train_dataset.class_to_idx)

with open(os.path.join(RESULTS_DIR, "class_mapping.json"), "w") as f:
    json.dump(train_dataset.class_to_idx, f, indent=4)

print("\nDataset Statistics")
print("-"*60)
print("Training Images  :", len(train_dataset))
print("Validation Images:", len(val_dataset))
print("Testing Images   :", len(test_dataset))
print("-"*60)

# ==========================================================
# DATALOADERS
# ==========================================================
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
print("Validation Batches :", len(val_loader))
print("Test Batches :", len(test_loader))

# ==========================================================
# TRAINING LOG
# ==========================================================

csv_path = os.path.join(RESULTS_DIR, "training_log.csv")
with open(csv_path, "w") as f:
    f.write("Epoch,TrainLoss,TrainAccuracy,ValidationLoss,ValidationAccuracy\n")

print("\nInitialization Completed Successfully.")
print("="*70)

# ==========================================================
# CREATE MODEL
# ==========================================================

model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)

print("\nLoading Domain TL Model...")
model.load_state_dict(torch.load(TL_MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
print("Domain TL Model Loaded Successfully.")

# ==========================================================
# FREEZE ALL PARAMETERS
# ==========================================================

for param in model.parameters():
    param.requires_grad = False

# ==========================================================
# UNFREEZE LAST BLOCKS (Block 3 Upwards)
# ==========================================================

for name, param in model.named_parameters():
    if (
        "blocks.3" in name or
        "blocks.4" in name or
        "blocks.5" in name or
        "blocks.6" in name or
        "conv_head" in name or
        "bn2" in name or
        "classifier" in name
    ):
        param.requires_grad = True
        print(f"Trainable : {name}")

# ==========================================================
# PARAMETER SUMMARY
# ==========================================================

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print("\nModel Summary")
print("-"*60)
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print(f"Trainable Percentage : {(100*trainable_params/total_params):.2f}%")
print("-"*60)

# ==========================================================
# LOSS, OPTIMIZER, SCHEDULER
# ==========================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.Adam([
    {
        'params': list(model.blocks[3].parameters()) +
                  list(model.blocks[4].parameters()) +
                  list(model.blocks[5].parameters()) +
                  list(model.conv_head.parameters()) +
                  list(model.bn2.parameters()),
        'lr': LR_BACKBONE
    },
    {
        'params': model.classifier.parameters(),
        'lr': LR_HEAD
    }
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

# ==========================================================
# HISTORY
# ==========================================================

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# KEEP ORIGINAL TRACKING AND EARLY STOP VARIABLES
best_val_loss = float('inf')
best_accuracy = 0.0
patience_counter = 0

best_weights = copy.deepcopy(model.state_dict())

print("\nStarting Domain Fine-Tuning...\n")

# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(NUM_EPOCHS):

    ##############################
    # TRAIN
    ##############################
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

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
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = running_correct / total

    ##############################
    # VALIDATION
    ##############################
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

    scheduler.step(val_loss)

    # Log Histories
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    # CSV entry
    with open(csv_path, "a") as f:
        f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

    print(
        f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
        f"Train Loss={train_loss:.4f} "
        f"Train Acc={train_acc:.4f} "
        f"Val Loss={val_loss:.4f} "
        f"Val Acc={val_acc:.4f}"
    )

    ##############################
    # SAVE BEST MODEL & EARLY STOP
    ##############################
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_accuracy = val_acc
        patience_counter = 0

        best_weights = copy.deepcopy(model.state_dict())
        # KEEP ORIGINAL FILENAME
        torch.save(
            best_weights,
            os.path.join(RESULTS_DIR, "best_domain_model.pth")
        )
        print(f"  --> Best model saved (Val Loss: {best_val_loss:.4f} | Val Acc: {best_accuracy:.4f})")
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs.")
        break

# Save backup architecture weights at final stop
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "last_domain_model.pth"))

print("\nFine-Tuning Completed Successfully.")
print(f"Best Validation Accuracy : {best_accuracy:.4f}")
print(f"Best Validation Loss     : {best_val_loss:.4f}")

# ==========================================================
# LOAD BEST MODEL FOR TESTING
# ==========================================================

print("\nLoading Best Domain Model...")
# KEEP ORIGINAL FILENAME
model.load_state_dict(
    torch.load(os.path.join(RESULTS_DIR, "best_domain_model.pth"), map_location=DEVICE)
)
model.eval()

# ==========================================================
# TESTING
# ==========================================================

all_labels = []
all_predictions = []
all_probabilities = []
all_paths = []
running_loss = 0.0

with torch.no_grad():
    sample_index = 0
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())

        for _ in range(images.size(0)):
            all_paths.append(test_dataset.samples[sample_index][0])
            sample_index += 1

test_loss = running_loss / len(test_dataset)

# ==========================================================
# METRICS
# ==========================================================

test_accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average="weighted")
recall = recall_score(all_labels, all_predictions, average="weighted")
f1 = f1_score(all_labels, all_predictions, average="weighted")

print("="*60)
print("TEST RESULTS")
print("="*60)
print(f"Loss      : {test_loss:.4f}")
print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

with open(os.path.join(RESULTS_DIR, "test_results.txt"), "w") as f:
    f.write("="*60 + "\n")
    f.write("DOMAIN FINE TUNING RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Loss      : {test_loss:.6f}\n")
    f.write(f"Accuracy  : {test_accuracy:.6f}\n")
    f.write(f"Precision : {precision:.6f}\n")
    f.write(f"Recall    : {recall:.6f}\n")
    f.write(f"F1 Score  : {f1:.6f}\n")

# ==========================================================
# CLASSIFICATION REPORT & PLOTS
# ==========================================================

report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes, digits=4)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
print(report)

cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap="Blues", values_format="d", colorbar=False, ax=ax)
plt.title("Domain FT Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# Save Prediction Records CSV 
prediction_rows = []
for i in range(len(all_labels)):
    prediction_rows.append({
        "Image": os.path.basename(all_paths[i]),
        "Image_Path": all_paths[i],
        "True_Label": test_dataset.classes[all_labels[i]],
        "Predicted_Label": test_dataset.classes[all_predictions[i]],
        "Confidence": float(np.max(all_probabilities[i]))
    })
pd.DataFrame(prediction_rows).to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

# Graph Generation Curves
plt.figure(figsize=(8, 6))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=300)
plt.close()

# Write Summary Text Document
with open(os.path.join(RESULTS_DIR, "training_summary.txt"), "w") as f:
    f.write("="*70+"\n")
    f.write("DOMAIN FINE TUNING SUMMARY\n")
    f.write("="*70+"\n\n")
    f.write(f"Model : {MODEL_NAME}\n")
    f.write(f"Epochs : {NUM_EPOCHS}\n")
    f.write(f"Backbone LR : {LR_BACKBONE}\n")
    f.write(f"Classifier LR : {LR_HEAD}\n")
    f.write(f"Batch Size : {BATCH_SIZE}\n")
    f.write(f"Image Size : {IMAGE_SIZE}\n\n")
    f.write(f"Training Images : {len(train_dataset)}\n")
    f.write(f"Validation Images : {len(val_dataset)}\n")
    f.write(f"Testing Images : {len(test_dataset)}\n\n")
    f.write("FINAL TEST RESULTS\n")
    f.write("-"*40+"\n")
    f.write(f"Loss : {test_loss:.6f}\n")
    f.write(f"Accuracy : {test_accuracy:.6f}\n")
    f.write(f"Precision : {precision:.6f}\n")
    f.write(f"Recall : {recall:.6f}\n")
    f.write(f"F1 Score : {f1:.6f}\n")

# ==========================================================
# SAVE BEST DOMAIN ENCODER
# (Used later in Fusion FT)
# ==========================================================

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

print("Best Domain Encoder Saved.")

# ==========================================================
# SAVE MISCLASSIFIED IMAGES 
# ==========================================================

for img_path, gt, pred in zip(all_paths, all_labels, all_predictions):
    if gt == pred:
        continue
    save_dir = os.path.join(MISCLASSIFIED_DIR, f"{test_dataset.classes[gt]}_as_{test_dataset.classes[pred]}")
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    img.save(os.path.join(save_dir, os.path.basename(img_path)))

print("\nDomain Fine-Tuning Completed Successfully.")
print(f"\nResults saved in:\n{RESULTS_DIR}")
