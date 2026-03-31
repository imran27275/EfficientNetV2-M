import os
import io
import random
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import torch.nn as nn

class RealFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
            real_dir = os.path.join(category_path, "0_real")
            fake_dir = os.path.join(category_path, "1_fake")
            if os.path.exists(real_dir):
                for img in os.listdir(real_dir):
                    self.samples.append((os.path.join(real_dir, img), 0))
            if os.path.exists(fake_dir):
                for img in os.listdir(fake_dir):
                    self.samples.append((os.path.join(fake_dir, img), 1))
        print("Total images: " + str(len(self.samples)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class RandomGaussianBlur:
    def __init__(self, prob=0.5, kernel_range=(3, 7)):
        self.prob = prob
        self.kernel_range = kernel_range
    def __call__(self, img):
        if random.random() < self.prob:
            k = random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))
            return transforms.functional.gaussian_blur(img, kernel_size=k)
        return img

class RandomJPEGCompression:
    def __init__(self, prob=0.5, quality_range=(30, 95)):
        self.prob = prob
        self.quality_range = quality_range
    def __call__(self, img):
        if random.random() < self.prob:
            quality = random.randint(self.quality_range[0], self.quality_range[1])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img

TRAIN_DIR      = "/home/woody/rlvl/rlvl153v/dataset/train"
VAL_DIR        = "/home/woody/rlvl/rlvl153v/dataset/val"
CHECKPOINT_DIR = "/home/woody/rlvl/rlvl153v/EfficientNet/checkpoints"
EPOCHS        = 20
BATCH_SIZE    = 256
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 16
MAX_GRAD_NORM = 1.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, filename):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "best_acc":  best_acc,
    }, filename)
    print("Checkpoint saved: " + filename)

def load_checkpoint(filename, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc    = checkpoint["best_acc"]
    print("Resumed from epoch " + str(checkpoint["epoch"] + 1))
    return start_epoch, best_acc

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("epoch_") and f.endswith(".pth")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.replace("epoch_", "").replace(".pth", "")))
    return os.path.join(checkpoint_dir, checkpoints[-1])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    RandomJPEGCompression(prob=0.5, quality_range=(30, 95)),
    RandomGaussianBlur(prob=0.5, kernel_range=(3, 7)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = RealFakeDataset(TRAIN_DIR, train_transform)
val_dataset   = RealFakeDataset(VAL_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))
model = timm.create_model("convnext_base", pretrained=True, num_classes=2)

    #   for ConvNeXt-B model use this code and remove the self.model code below
    #   model = timm.create_model("tf_efficientnetv2_m", pretrained=True, num_classes=2)

model = timm.create_model("tf_efficientnetv2_m", pretrained=True, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler    = GradScaler()
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

start_epoch = 0
best_acc    = 0.0
latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
if latest_ckpt:
    print("Found checkpoint: " + latest_ckpt)
    start_epoch, best_acc = load_checkpoint(latest_ckpt, model, optimizer, scheduler, scaler)
else:
    print("No checkpoint found - starting from scratch.")

for epoch in range(start_epoch, EPOCHS):
    print("\nEpoch " + str(epoch + 1) + "/" + str(EPOCHS))
    model.train()
    correct      = 0
    total        = 0
    running_loss = 0.0
    loop = tqdm(train_loader, desc="Training")
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        preds         = torch.argmax(outputs, dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)
        running_loss += loss.item()
        loop.set_postfix(loss=round(loss.item(), 4), acc=round(correct / total, 4))
    train_acc  = correct / total
    train_loss = running_loss / len(train_loader)
    print("Train - loss: " + str(round(train_loss, 4)) + " acc: " + str(round(train_acc, 4)))
    model.eval()
    correct  = 0
    total    = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            preds     = torch.argmax(outputs, dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)
            val_loss += loss.item()
    val_acc  = correct / total
    val_loss = val_loss / len(val_loader)
    print("Val - loss: " + str(round(val_loss, 4)) + " acc: " + str(round(val_acc, 4)))
    scheduler.step()
    save_checkpoint(epoch, model, optimizer, scheduler, scaler, best_acc, os.path.join(CHECKPOINT_DIR, "epoch_" + str(epoch) + ".pth"))
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print("Best model updated! val_acc=" + str(round(best_acc, 4)))

print("Training complete. Best val accuracy: " + str(round(best_acc, 4)))
