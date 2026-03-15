import os
import random
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


# ==========================
# Config
# ==========================

TEST_DIR       = "/dataset/test"
CHECKPOINT_DIR = "/checkpoints"
BEST_MODEL     = "/checkpoints/best_model.pth"
BATCH_SIZE     = 256
NUM_WORKERS    = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Real", "Fake"]


# ==========================
# Dataset — single generator
# ==========================

class GeneratorDataset(Dataset):
    """
    Loads images from a generator folder.
    Supports two structures automatically:

    Flat (StyleGAN, BigGAN etc.):
      generator/
        0_real/
        1_fake/

    Nested/categorized (ProGAN):
      generator/
        airplane/
          0_real/
          1_fake/
        bicycle/
          0_real/
          1_fake/
    """

    def __init__(self, generator_dir, transform=None):

        self.samples   = []
        self.transform = transform

        # Try flat structure first (0_real / 1_fake directly inside)
        real_dir = os.path.join(generator_dir, "0_real")
        fake_dir = os.path.join(generator_dir, "1_fake")

        if os.path.exists(real_dir) or os.path.exists(fake_dir):
            # Flat structure
            self._load_dir(real_dir, label=0)
            self._load_dir(fake_dir, label=1)
        else:
            # Nested structure — loop through category subfolders
            for category in sorted(os.listdir(generator_dir)):
                category_path = os.path.join(generator_dir, category)
                if not os.path.isdir(category_path):
                    continue
                self._load_dir(os.path.join(category_path, "0_real"), label=0)
                self._load_dir(os.path.join(category_path, "1_fake"), label=1)

    def _load_dir(self, path, label):
        if not os.path.exists(path):
            return
        for img in os.listdir(path):
            if img.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.samples.append((os.path.join(path, img), label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================
# Blur + JPEG helpers (same classes as trainer.py)
# ==========================

class RandomGaussianBlur:
    """Apply Gaussian blur with given probability."""
    def __init__(self, prob=0.5, kernel_range=(3, 7)):
        self.prob         = prob
        self.kernel_range = kernel_range

    def __call__(self, img):
        if random.random() < self.prob:
            k = random.choice(range(self.kernel_range[0],
                                    self.kernel_range[1] + 1, 2))
            return transforms.functional.gaussian_blur(img, kernel_size=k)
        return img


class RandomJPEGCompression:
    """Apply JPEG compression with given probability."""
    def __init__(self, prob=0.5, quality_range=(30, 95)):
        self.prob          = prob
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() < self.prob:
            import io
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img


# ==========================
# Transforms
# ==========================

# Clean transform — no augmentation (standard evaluation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Augmented transform — simulates real-world degraded images
# Use this to test robustness against blur/compression artifacts
test_transform_augmented = transforms.Compose([
    transforms.Resize((224, 224)),
    RandomJPEGCompression(prob=0.5, quality_range=(30, 95)),
    RandomGaussianBlur(prob=0.5, kernel_range=(3, 7)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Set which transform to use ──
# USE_AUGMENTED_TEST = False  → clean evaluation (recommended first)
# USE_AUGMENTED_TEST = True   → robustness evaluation under blur+JPEG
USE_AUGMENTED_TEST = False

active_transform = test_transform_augmented if USE_AUGMENTED_TEST else test_transform
mode_label       = "Augmented (Blur+JPEG 0.1)" if USE_AUGMENTED_TEST else "Clean (no augmentation)"
print(f"Test mode: {mode_label}")


# ==========================
# Load Model
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists(BEST_MODEL):
    raise FileNotFoundError(
        f"No best_model.pth found at '{BEST_MODEL}'. "
        "Please train the model first."
    )

print("Loading EfficientNetV2-M...")
model = timm.create_model(
    "tf_efficientnetv2_m",
    pretrained=False,
    num_classes=2
)
model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded from: {BEST_MODEL}\n")


# ==========================
# Evaluate one generator
# ==========================

def evaluate_generator(generator_name, generator_dir):
    """
    Run inference on one generator folder.
    Returns a dict of metrics, or None if folder is empty/missing.
    """
    dataset = GeneratorDataset(generator_dir, active_transform)

    if len(dataset) == 0:
        print(f"  ⚠  Skipping '{generator_name}' — no images found.")
        return None

    n_real = sum(1 for _, l in dataset.samples if l == 0)
    n_fake = sum(1 for _, l in dataset.samples if l == 1)
    print(f"  Images — Real: {n_real}  Fake: {n_fake}  Total: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  Evaluating", leave=False):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of Fake

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # Guard: AP and AUC need both classes present
    has_both_classes = len(np.unique(all_labels)) == 2

    accuracy      = accuracy_score(all_labels, all_preds)
    avg_precision = average_precision_score(all_labels, all_probs) if has_both_classes else float("nan")
    roc_auc       = roc_auc_score(all_labels, all_probs)           if has_both_classes else float("nan")
    conf_mat      = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    report        = classification_report(
                        all_labels, all_preds,
                        target_names=CLASS_NAMES,
                        zero_division=0
                    )

    tn, fp, fn, tp = conf_mat.ravel()

    return {
        "accuracy":      accuracy,
        "avg_precision": avg_precision,
        "roc_auc":       roc_auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "report":        report,
        "n_total":       len(dataset),
        "n_real":        n_real,
        "n_fake":        n_fake,
    }


# ==========================
# Run across all generators
# ==========================

# Auto-detect generator folders inside TEST_DIR
generators = sorted([
    d for d in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, d))
])

if not generators:
    raise RuntimeError(f"No subfolders found in TEST_DIR: {TEST_DIR}")

print(f"Found {len(generators)} generator(s): {generators}\n")

results = {}

for gen in generators:
    gen_dir = os.path.join(TEST_DIR, gen)
    print(f"{'='*55}")
    print(f"  Generator: {gen.upper()}")
    metrics = evaluate_generator(gen, gen_dir)
    if metrics:
        results[gen] = metrics


# ==========================
# Print Summary Table
# ==========================

print(f"\n{'='*55}")
print("         CROSS-GENERATOR EVALUATION SUMMARY")
print(f"{'='*55}")
print(f"{'Generator':<15} {'Images':>7} {'Accuracy':>10} {'Avg Prec':>10} {'ROC-AUC':>10}")
print(f"{'-'*55}")

for gen, m in results.items():
    ap  = f"{m['avg_precision']:.4f}" if not np.isnan(m['avg_precision']) else "  N/A  "
    auc = f"{m['roc_auc']:.4f}"       if not np.isnan(m['roc_auc'])       else "  N/A  "
    print(f"{gen:<15} {m['n_total']:>7} {m['accuracy']*100:>9.2f}% {ap:>10} {auc:>10}")

print(f"{'='*55}")

# Overall average across all generators
all_acc = [m["accuracy"]      for m in results.values()]
all_ap  = [m["avg_precision"] for m in results.values() if not np.isnan(m["avg_precision"])]
all_auc = [m["roc_auc"]       for m in results.values() if not np.isnan(m["roc_auc"])]

print(f"{'AVERAGE':<15} {'':>7} {np.mean(all_acc)*100:>9.2f}% "
      f"{np.mean(all_ap):>10.4f} {np.mean(all_auc):>10.4f}")
print(f"{'='*55}")


# ==========================
# Print Per-Generator Detail
# ==========================

print("\n\nDETAILED RESULTS PER GENERATOR")

for gen, m in results.items():
    print(f"\n{'='*55}")
    print(f"  {gen.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy          : {m['accuracy']*100:.2f}%")
    print(f"  Average Precision : {m['avg_precision']:.4f}" if not np.isnan(m['avg_precision']) else "  Average Precision : N/A")
    print(f"  ROC-AUC           : {m['roc_auc']:.4f}"       if not np.isnan(m['roc_auc'])       else "  ROC-AUC           : N/A")
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted Real   Predicted Fake")
    print(f"    Actual Real     {m['tn']:<16}   {m['fp']}")
    print(f"    Actual Fake     {m['fn']:<16}   {m['tp']}")
    print(f"\n  True Positives  (Fake correctly detected) : {m['tp']}")
    print(f"  True Negatives  (Real correctly detected) : {m['tn']}")
    print(f"  False Positives (Real misclassified Fake) : {m['fp']}")
    print(f"  False Negatives (Fake misclassified Real) : {m['fn']}")
    print(f"\n  Classification Report:")
    print(m['report'])