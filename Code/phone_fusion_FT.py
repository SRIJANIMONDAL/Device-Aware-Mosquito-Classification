import os
import json
import random
import warnings
import copy

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

NUM_CLASSES = 4
DOMAIN_CLASSES = 3

NUM_EPOCHS = 100
PATIENCE = 15

MODEL_NAME = "tf_efficientnetv2_b0"

# Layer-wise learning rate
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = r"/mnt/d/Projects/Mosquito Analysis"
BASE_1_DIR=r"/mnt/d/Projects/Mosquito Analysis/Results Improved Finetune with Background"

# Dataset
TRAIN_DIR = os.path.join(
    BASE_DIR,
    "body_images_devices_merged",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "body_images_devices_merged",
    "val"
)

TEST_MACRO_DIR = os.path.join(
    BASE_DIR,
    "body_images_devices_merged",
    "test_macro"
)

TEST_MICRO_DIR = os.path.join(
    BASE_DIR,
    "body_images_devices_merged",
    "test_micro"
)

TEST_PHONE_DIR = os.path.join(
    BASE_DIR,
    "body_images_devices_merged",
    "test_phone"
)


# ==========================================================
# PRETRAINED MODELS
# ==========================================================

# Species FT model
SPECIES_FT_PATH = os.path.join(
    BASE_1_DIR,
    "results_phone_finetune_new",
    "best_phone_finetune_new_model.pth"
)

# Domain FT model
DOMAIN_FT_PATH = os.path.join(
    BASE_DIR,
    "domain_FT",
    "best_domain_model.pth"
)

# Output Directories
RESULTS_DIR = os.path.join(
    BASE_DIR,
    "phone_fusion_FT"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

MISCLASSIFIED_DIR = os.path.join(
    RESULTS_DIR,
    "misclassified_images"
)
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PIN_MEMORY = torch.cuda.is_available()

print("="*70)
print("DEVICE :", DEVICE)
print("PIN MEMORY :", PIN_MEMORY)
print("="*70)


# ==========================================================
# TRANSFORMS
# ==========================================================

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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# DATASET
# ==========================================================

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = ImageFolder(VAL_DIR, transform=test_transform)

macro_dataset = ImageFolder(TEST_MACRO_DIR, transform=test_transform)
micro_dataset = ImageFolder(TEST_MICRO_DIR, transform=test_transform)
phone_dataset = ImageFolder(TEST_PHONE_DIR, transform=test_transform)

print("\nClass Mapping")
print(train_dataset.class_to_idx)

with open(os.path.join(RESULTS_DIR, "class_mapping.json"), "w") as f:
    json.dump(train_dataset.class_to_idx, f, indent=4)

print("\nDataset Statistics")
print("-"*60)
print("Training Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))
print("Macro Test :", len(macro_dataset))
print("Micro Test :", len(micro_dataset))
print("Phone Test :", len(phone_dataset))
print("-"*60)


# ==========================================================
# DATALOADERS
# ==========================================================

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY
)
macro_loader = DataLoader(
    macro_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY
)
micro_loader = DataLoader(
    micro_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY
)
phone_loader = DataLoader(
    phone_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY
)

print("\nFusion FT Initialization Completed")
print("="*70)


# ==========================================================
# FUSION NETWORK DEFINITION FOR FUSION FINE-TUNING
# ==========================================================

class FusionNet_FT(nn.Module):

    def __init__(self):
        super().__init__()

        # ==================================================
        # 1. Species Encoder θ1
        # ==================================================
        self.species_encoder = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            drop_rate=0.5
        )

        # Load Species FT weights (filtering out classifier/head keys)
        species_checkpoint = torch.load(SPECIES_FT_PATH, map_location=DEVICE)
        species_checkpoint = {
            k: v for k, v in species_checkpoint.items()
            if not (k.startswith("classifier") or k.startswith("head"))
        }
        self.species_encoder.load_state_dict(species_checkpoint, strict=False)

        # ==================================================
        # 2. Domain Encoder θ2
        # ==================================================
        self.domain_encoder = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            drop_rate=0.3
        )

        # Load Domain FT weights (filtering out classifier/head keys)
        domain_checkpoint = torch.load(DOMAIN_FT_PATH, map_location=DEVICE)
        domain_checkpoint = {
            k: v for k, v in domain_checkpoint.items()
            if not (k.startswith("classifier") or k.startswith("head"))
        }
        self.domain_encoder.load_state_dict(domain_checkpoint, strict=False)

        # Dynamically calculate input features from backbones
        in_features = self.species_encoder.num_features + self.domain_encoder.num_features

        # ==================================================
        # 3. Fusion Classifier (Initialized Fresh)
        # ==================================================
        self.fusion_classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

        # ==================================================
        # 4. FREEZE DOMAIN ENCODER θ2
        # ==================================================
        for param in self.domain_encoder.parameters():
            param.requires_grad = False

        self.domain_encoder.eval()

        # ==================================================
        # 5. FREEZE SPECIES ENCODER FIRST
        # ==================================================
        for param in self.species_encoder.parameters():
            param.requires_grad = False

        # ==================================================
        # 6. UNFREEZE LAST 50% SPECIES BLOCKS & HEAD
        # ==================================================
        for param in self.species_encoder.blocks[3:].parameters():
            param.requires_grad = True

        for param in self.species_encoder.conv_head.parameters():
            param.requires_grad = True

        for param in self.species_encoder.bn2.parameters():
            param.requires_grad = True

        # ==================================================
        # 7. Fusion Classifier Trainable
        # ==================================================
        for param in self.fusion_classifier.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        """Keep domain encoder strictly in eval mode during training."""
        super().train(mode)
        self.domain_encoder.eval()
        return self

    def forward(self, x):
        # Species feature
        species_feature = self.species_encoder(x)

        # Domain feature frozen
        with torch.no_grad():
            domain_feature = self.domain_encoder(x)

        # Concatenate
        fusion_feature = torch.cat((species_feature, domain_feature), dim=1)

        # Classification output
        return self.fusion_classifier(fusion_feature)


# ==========================================================
# MODEL INITIALIZATION
# ==========================================================

model = FusionNet_FT().to(DEVICE)

print("\nFusion FT Model Created Successfully")


# ==========================================================
# PARAMETER CHECK
# ==========================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Parameters")
print("-"*60)
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print(f"Trainable Percentage : {100*trainable_params/total_params:.2f}%")
print("-"*60)


# ==========================================================
# LOSS FUNCTION
# ==========================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.AdamW(
    [
        {
            # Species encoder fine-tuning layers
            "params": (
                list(model.species_encoder.blocks[3:].parameters())
                + list(model.species_encoder.conv_head.parameters())
                + list(model.species_encoder.bn2.parameters())
            ),
            "lr": LR_BACKBONE
        },
        {
            # New fusion classifier
            "params": model.fusion_classifier.parameters(),
            "lr": LR_HEAD
        }
    ]
)


# ==========================================================
# LEARNING RATE SCHEDULER
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# ==========================================================
# HISTORY & CSV LOG
# ==========================================================

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

csv_path = os.path.join(RESULTS_DIR, "training_log.csv")

with open(csv_path, "w") as f:
    f.write("Epoch,TrainLoss,TrainAccuracy,ValidationLoss,ValidationAccuracy\n")


# ==========================================================
# TRAINING LOOP
# ==========================================================

best_val_loss = float("inf")
best_val_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())
patience_counter = 0

print("\nStarting Fusion Fine-Tuning...\n")

for epoch in range(NUM_EPOCHS):

    # --- TRAIN ---
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

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            val_loss += loss.item() * images.size(0)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    # Scheduler Step
    scheduler.step(val_loss)

    # Save History
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    with open(csv_path, "a") as f:
        f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

    print(
        f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # --- SAVE BEST MODEL ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_acc
        patience_counter = 0

        best_weights = copy.deepcopy(model.state_dict())
        torch.save(
            best_weights,
            os.path.join(RESULTS_DIR, "best_fusion_FT_model.pth")
        )
        print(
            f"--> Best Fusion FT Model Saved "
            f"(Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_accuracy:.4f})"
        )
    else:
        patience_counter += 1

    # Early Stopping
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break


# Save Last Model
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "last_fusion_FT_model.pth"))

print("\nFusion Fine-Tuning Completed")
print(f"Best Validation Loss : {best_val_loss:.4f}")
print(f"Best Validation Accuracy : {best_val_accuracy:.4f}")


# ==========================================================
# LOAD BEST FUSION FT MODEL
# ==========================================================

print("\nLoading Best Fusion FT Model...")
model.load_state_dict(
    torch.load(os.path.join(RESULTS_DIR, "best_fusion_FT_model.pth"), map_location=DEVICE)
)
model.eval()
print("Best Fusion FT Model Loaded Successfully")


# Save Final Species Encoder
FUSION_FT_ENCODER_PATH = os.path.join(RESULTS_DIR, "best_species_encoder_fusion_FT.pth")
torch.save(model.species_encoder.state_dict(), FUSION_FT_ENCODER_PATH)
print("\nFinal Species Encoder Saved:", FUSION_FT_ENCODER_PATH)


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_dataset(dataloader, dataset, name_prefix):
    print("\n" + "="*70)
    print(f"Evaluating Fusion FT : {name_prefix.upper()}")
    print("="*70)

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []
    running_loss = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
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

            batch_start = i * dataloader.batch_size
            for idx in range(images.size(0)):
                all_paths.append(dataset.samples[batch_start + idx][0])

    test_loss = running_loss / len(dataset)
    test_accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    print(
        f"\nLoss      : {test_loss:.4f}\n"
        f"Accuracy  : {test_accuracy:.4f}\n"
        f"Precision : {precision:.4f}\n"
        f"Recall    : {recall:.4f}\n"
        f"F1 Score  : {f1:.4f}\n"
    )

    # Save Text Results
    with open(os.path.join(RESULTS_DIR, f"{name_prefix}_test_results.txt"), "w") as f:
        f.write("FUSION FINE-TUNING RESULTS\n" + "="*50 + "\n\n")
        f.write(f"Loss      : {test_loss:.6f}\n")
        f.write(f"Accuracy  : {test_accuracy:.6f}\n")
        f.write(f"Precision : {precision:.6f}\n")
        f.write(f"Recall    : {recall:.6f}\n")
        f.write(f"F1 Score  : {f1:.6f}\n")

    # Save Classification Report
    report = classification_report(
        all_labels, all_predictions, target_names=dataset.classes, digits=4
    )
    with open(os.path.join(RESULTS_DIR, f"{name_prefix}_classification_report.txt"), "w") as f:
        f.write(report)

    # Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", values_format="d", colorbar=False, ax=ax)
    plt.title(f"Fusion FT Confusion Matrix - {name_prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name_prefix}_confusion_matrix.png"), dpi=300)
    plt.close()

    # Save Prediction CSV
    prediction_rows = []
    for i in range(len(all_labels)):
        prediction_rows.append({
            "Image": os.path.basename(all_paths[i]),
            "Image_Path": all_paths[i],
            "True_Label": dataset.classes[all_labels[i]],
            "Predicted_Label": dataset.classes[all_predictions[i]],
            "Confidence": float(np.max(all_probabilities[i]))
        })

    pd.DataFrame(prediction_rows).to_csv(
        os.path.join(RESULTS_DIR, f"{name_prefix}_predictions.csv"), index=False
    )

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================================
# RUN ALL DOMAIN TESTS
# ==========================================================

macro_metrics = evaluate_dataset(macro_loader, macro_dataset, "macro")
micro_metrics = evaluate_dataset(micro_loader, micro_dataset, "micro")
phone_metrics = evaluate_dataset(phone_loader, phone_dataset, "phone")


# ==========================================================
# SAVE TRAINING CURVES
# ==========================================================

plt.figure(figsize=(8, 5))
plt.plot(train_acc_history, label="Train")
plt.plot(val_acc_history, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fusion FT Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train")
plt.plot(val_loss_history, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fusion FT Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300)
plt.close()


# ==========================================================
# FINAL SUMMARY
# ==========================================================

summary_path = os.path.join(RESULTS_DIR, "training_summary.txt")

with open(summary_path, "w") as f:
    f.write("="*70 + "\nFUSION FINE-TUNING SUMMARY\n" + "="*70 + "\n\n")
    f.write(f"Model : {MODEL_NAME}\n")
    f.write(f"Epochs : {NUM_EPOCHS}\n")
    f.write(f"Batch Size : {BATCH_SIZE}\n")
    f.write(f"Image Size : {IMAGE_SIZE}\n\n")
    f.write("FINAL TEST RESULTS\n" + "-"*50 + "\n")

    for name, m in [("Macro", macro_metrics), ("Micro", micro_metrics), ("Phone", phone_metrics)]:
        f.write(f"\n{name} TEST\n")
        f.write(f"Accuracy : {m['accuracy']:.6f}\n")
        f.write(f"Precision : {m['precision']:.6f}\n")
        f.write(f"Recall : {m['recall']:.6f}\n")
        f.write(f"F1 Score : {m['f1']:.6f}\n")

print("\nFusion FT Pipeline Completed Successfully")
print(f"All results saved at:\n{RESULTS_DIR}")