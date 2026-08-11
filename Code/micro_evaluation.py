import os
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# CONFIG
# =========================================================

IMAGE_SIZE = 300
BATCH_SIZE = 16
NUM_CLASSES = 4
MODEL_NAME = "tf_efficientnetv2_b0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# PATHS
# =========================================================

BASE_DIR = r"/mnt/d/Projects/Mosquito Analysis"

RESULTS_DIR = os.path.join(BASE_DIR, "micro_evaluation")
os.makedirs(RESULTS_DIR, exist_ok=True)

FUSION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "micro_fusion_FT",
    "best_fusion_FT_model.pth"
)

TEST_MACRO_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "test_macro")
TEST_MICRO_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "test_micro")
TEST_PHONE_DIR = os.path.join(BASE_DIR, "body_images_devices_merged", "test_phone")

# =========================================================
# TRANSFORM
# =========================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DATASETS
# =========================================================

macro_dataset = ImageFolder(TEST_MACRO_DIR, transform=test_transform)
micro_dataset = ImageFolder(TEST_MICRO_DIR, transform=test_transform)
phone_dataset = ImageFolder(TEST_PHONE_DIR, transform=test_transform)

macro_loader = DataLoader(macro_dataset, batch_size=BATCH_SIZE, shuffle=False)
micro_loader = DataLoader(micro_dataset, batch_size=BATCH_SIZE, shuffle=False)
phone_loader = DataLoader(phone_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# ORIGINAL FUSION ARCHITECTURE
# =========================================================

class FusionNetInference(nn.Module):
    def __init__(self):
        super().__init__()

        # species encoder
        self.species_encoder = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0
        )

        # domain encoder (needed only to load checkpoint)
        self.domain_encoder = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0
        )

        in_features = (
            self.species_encoder.num_features +
            self.domain_encoder.num_features
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        # -------- species feature --------
        species_feature = self.species_encoder(x)

        # -------- REMOVE DOMAIN ENCODER --------
        # Instead of: domain_feature = self.domain_encoder(x)
        # use zero vector of same shape
        domain_feature = torch.zeros_like(species_feature)

        fusion_feature = torch.cat((species_feature, domain_feature), dim=1)

        return self.fusion_classifier(fusion_feature)


# =========================================================
# LOAD TRAINED FUSION MODEL
# =========================================================

model = FusionNetInference().to(DEVICE)

checkpoint = torch.load(FUSION_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)

model.eval()

print("Loaded trained fusion model")
print("Domain encoder removed during inference")

# =========================================================
# EVALUATION FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()


def evaluate_dataset(dataloader, dataset, name):

    print(f"\n{'='*60}\n{name.upper()} TEST\n{'='*60}")

    all_labels = []
    all_preds = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    loss = running_loss / len(dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Loss      : {loss:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=dataset.classes,
        digits=4
    )

    with open(os.path.join(RESULTS_DIR, f"{name}_classification_report.txt"), "w") as f:
        f.write(report)

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=dataset.classes)

    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(cmap="Blues", values_format="d", colorbar=False, ax=ax)
    plt.title(f"Without Domain Encoder - {name}")
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.png"), dpi=300)
    plt.close()

    # save predictions
    pd.DataFrame({
        "True": [dataset.classes[i] for i in all_labels],
        "Predicted": [dataset.classes[i] for i in all_preds]
    }).to_csv(
        os.path.join(RESULTS_DIR, f"{name}_predictions.csv"),
        index=False
    )


# =========================================================
# RUN TESTS
# =========================================================

evaluate_dataset(macro_loader, macro_dataset, "macro")
evaluate_dataset(micro_loader, micro_dataset, "micro")
evaluate_dataset(phone_loader, phone_dataset, "phone")

print("\nInference completed.")
print("Results saved to:", RESULTS_DIR)
