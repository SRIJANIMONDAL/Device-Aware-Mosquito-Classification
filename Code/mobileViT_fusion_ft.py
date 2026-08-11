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

MODEL_NAME = "mobilevit_s"

# Layer-wise learning rate
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3

BASE_DIR = r"/mnt/d/Projects/Mosquito Analysis"

# Common Dataset Paths
TRAIN_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "train")
VAL_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "val")

TEST_MACRO_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "test_macro")
TEST_MICRO_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "test_micro")
TEST_PHONE_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "test_phone")

# Shared Domain Encoder Path (Updated to MobileViT Domain FT)
DOMAIN_FT_PATH = os.path.join(
    BASE_DIR,
    "mobileViT_domain_FT",
    "best_mobilevit_model.pth"
)

# Configuration array for executing Macro, Micro, and Phone Fusion FT pipelines
DOMAINS = [
    {
        "name": "macro",
        "species_path": os.path.join(BASE_DIR, "results_mobilevit_macro_train", "best_mobilevit_macro_model.pth"),
        "results_dir": os.path.join(BASE_DIR, "macro_mobilevit_fusion_FT")
    },
    {
        "name": "micro",
        "species_path": os.path.join(BASE_DIR, "results_mobilevit_micro_train", "best_mobilevit_micro_model.pth"),
        "results_dir": os.path.join(BASE_DIR, "micro_mobilevit_fusion_FT")
    },
    {
        "name": "phone",
        "species_path": os.path.join(BASE_DIR, "results_mobilevit_phone_train", "best_mobilevit_phone_model.pth"),
        "results_dir": os.path.join(BASE_DIR, "phone_mobilevit_fusion_FT")
    }
]

# Standard Transforms
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
# FUSION NETWORK DEFINITION FOR MOBILEVIT-S FINE-TUNING
# ==========================================================

class MobileViTFusionNet_FT(nn.Module):

    def __init__(self, species_ft_path, domain_ft_path, device):
        super().__init__()

        # 1. Species Encoder θ1
        self.species_encoder = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            drop_rate=0.5
        )

        # Load Species FT weights (filter classifier/head/fc keys)
        species_checkpoint = torch.load(species_ft_path, map_location=device)
        species_checkpoint = {
            k: v for k, v in species_checkpoint.items()
            if not (k.startswith("classifier") or k.startswith("head") or k.startswith("fc"))
        }
        self.species_encoder.load_state_dict(species_checkpoint, strict=False)

        # 2. Domain Encoder θ2
        self.domain_encoder = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,
            drop_rate=0.3
        )

        # Load Domain FT weights (filter classifier/head/fc keys)
        domain_checkpoint = torch.load(domain_ft_path, map_location=device)
        domain_checkpoint = {
            k: v for k, v in domain_checkpoint.items()
            if not (k.startswith("classifier") or k.startswith("head") or k.startswith("fc"))
        }
        self.domain_encoder.load_state_dict(domain_checkpoint, strict=False)

        # Input feature projection
        in_features = self.species_encoder.num_features + self.domain_encoder.num_features

        # 3. Fusion Classifier
        self.fusion_classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

        # 4. Freeze Domain Encoder θ2
        for param in self.domain_encoder.parameters():
            param.requires_grad = False
        self.domain_encoder.eval()

        # 5. Freeze Species Encoder initial layers
        for param in self.species_encoder.parameters():
            param.requires_grad = False

        # 6. Unfreeze last MobileViT stages (stages.3 & stages.4)
        for name, param in self.species_encoder.named_parameters():
            if "stages.3" in name or "stages.4" in name:
                param.requires_grad = True

        # 7. Fusion Classifier Trainable
        for param in self.fusion_classifier.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        """Keep domain encoder strictly in eval mode during training."""
        super().train(mode)
        self.domain_encoder.eval()
        return self

    def forward(self, x):
        species_feature = self.species_encoder(x)
        with torch.no_grad():
            domain_feature = self.domain_encoder(x)

        fusion_feature = torch.cat((species_feature, domain_feature), dim=1)
        return self.fusion_classifier(fusion_feature)


# ==========================================================
# EVALUATION FUNCTION
# ==========================================================

def evaluate_dataset(model, dataloader, dataset, name_prefix, results_dir, criterion, device):
    print("\n" + "="*70)
    print(f"Evaluating MobileViT Fusion FT : {name_prefix.upper()}")
    print("="*70)

    all_labels, all_predictions, all_probabilities, all_paths = [], [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

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

    with open(os.path.join(results_dir, f"{name_prefix}_test_results.txt"), "w") as f:
        f.write("MOBILEVIT FUSION FINE-TUNING RESULTS\n" + "="*50 + "\n\n")
        f.write(f"Loss      : {test_loss:.6f}\n")
        f.write(f"Accuracy  : {test_accuracy:.6f}\n")
        f.write(f"Precision : {precision:.6f}\n")
        f.write(f"Recall    : {recall:.6f}\n")
        f.write(f"F1 Score  : {f1:.6f}\n")

    report = classification_report(all_labels, all_predictions, target_names=dataset.classes, digits=4)
    with open(os.path.join(results_dir, f"{name_prefix}_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", values_format="d", colorbar=False, ax=ax)
    plt.title(f"MobileViT Fusion FT Confusion Matrix - {name_prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{name_prefix}_confusion_matrix.png"), dpi=300)
    plt.close()

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
        os.path.join(results_dir, f"{name_prefix}_predictions.csv"), index=False
    )

    return {"loss": test_loss, "accuracy": test_accuracy, "precision": precision, "recall": recall, "f1": f1}


# ==========================================================
# PIPELINE EXECUTION FUNCTION
# ==========================================================

def run_fusion_pipeline(domain_info):
    domain_name = domain_info["name"]
    species_ft_path = domain_info["species_path"]
    results_dir = domain_info["results_dir"]

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "misclassified_images"), exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    print("\n" + "="*70)
    print(f"STARTING PIPELINE: {domain_name.upper()} MOBILEVIT FUSION FT")
    print(f"DEVICE: {device} | PIN MEMORY: {pin_memory}")
    print("="*70)

    # Datasets
    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = ImageFolder(VAL_DIR, transform=test_transform)
    macro_dataset = ImageFolder(TEST_MACRO_DIR, transform=test_transform)
    micro_dataset = ImageFolder(TEST_MICRO_DIR, transform=test_transform)
    phone_dataset = ImageFolder(TEST_PHONE_DIR, transform=test_transform)

    with open(os.path.join(results_dir, "class_mapping.json"), "w") as f:
        json.dump(train_dataset.class_to_idx, f, indent=4)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    macro_loader = DataLoader(macro_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    micro_loader = DataLoader(micro_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    phone_loader = DataLoader(phone_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # Model Initialization
    model = MobileViTFusionNet_FT(species_ft_path, DOMAIN_FT_PATH, device).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Trainable Percentage : {100*trainable_params/total_params:.2f}%")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer (MobileViT stages.3 & stages.4 + Fusion Classifier)
    species_backbone_params = [
        p for n, p in model.species_encoder.named_parameters()
        if ("stages.3" in n or "stages.4" in n) and p.requires_grad
    ]
    
    optimizer = torch.optim.AdamW(
        [
            {"params": species_backbone_params, "lr": LR_BACKBONE},
            {"params": model.fusion_classifier.parameters(), "lr": LR_HEAD}
        ]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history   = [], []

    csv_path = os.path.join(results_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("Epoch,TrainLoss,TrainAccuracy,ValidationLoss,ValidationAccuracy\n")

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    patience_counter = 0

    print(f"\nTraining {domain_name.upper()} MobileViT Fusion FT Network...\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

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
        train_acc  = running_correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * images.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total

        scheduler.step(val_loss)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        with open(csv_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

        print(
            f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(results_dir, "best_fusion_FT_model.pth"))
            print(f"--> Best {domain_name.upper()} Model Saved (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_accuracy:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    # Save Last Model
    torch.save(model.state_dict(), os.path.join(results_dir, "last_fusion_FT_model.pth"))

    # Load Best Weights for Evaluation
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_fusion_FT_model.pth"), map_location=device))
    model.eval()

    # Save Species Encoder
    torch.save(model.species_encoder.state_dict(), os.path.join(results_dir, "best_species_encoder_fusion_FT.pth"))

    # Evaluate across all domains
    macro_m = evaluate_dataset(model, macro_loader, macro_dataset, "macro", results_dir, criterion, device)
    micro_m = evaluate_dataset(model, micro_loader, micro_dataset, "micro", results_dir, criterion, device)
    phone_m = evaluate_dataset(model, phone_loader, phone_dataset, "phone", results_dir, criterion, device)

    # Save Curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_history, label="Train")
    plt.plot(val_acc_history, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{domain_name.upper()} MobileViT Fusion FT Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Train")
    plt.plot(val_loss_history, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{domain_name.upper()} MobileViT Fusion FT Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Final Summary File
    with open(os.path.join(results_dir, "training_summary.txt"), "w") as f:
        f.write("="*70 + f"\n{domain_name.upper()} MOBILEVIT FUSION FINE-TUNING SUMMARY\n" + "="*70 + "\n\n")
        f.write(f"Model : {MODEL_NAME}\nEpochs : {NUM_EPOCHS}\nBatch Size : {BATCH_SIZE}\nImage Size : {IMAGE_SIZE}\n\n")
        f.write("FINAL TEST RESULTS\n" + "-"*50 + "\n")
        for name, m in [("Macro", macro_m), ("Micro", micro_m), ("Phone", phone_m)]:
            f.write(f"\n{name} TEST\n")
            f.write(f"Accuracy : {m['accuracy']:.6f}\n")
            f.write(f"Precision : {m['precision']:.6f}\n")
            f.write(f"Recall : {m['recall']:.6f}\n")
            f.write(f"F1 Score : {m['f1']:.6f}\n")


if __name__ == "__main__":
    for domain in DOMAINS:
        run_fusion_pipeline(domain)