import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import time
import random
import cv2
from sklearn.model_selection import train_test_split
from collections import deque

def create_datasets(blurred_path, origin_path, train_ratio=0.8, image_size=(240, 320), cache_size=1000, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    blurred_files = set(os.listdir(blurred_path))
    origin_files = set(os.listdir(origin_path))
    common_files = [i for i in blurred_files.intersection(origin_files) if i.endswith('.jpg')]

    train_files, validate_files = train_test_split(common_files, train_size=train_ratio, random_state=random_seed, shuffle=True)

    train_dataset = FaceDeblurDataset(
        blurred_path,
        origin_path,
        image_size,
        cache_size,
        True,
        train_files
    )

    validate_dataset = FaceDeblurDataset(
        blurred_path,
        origin_path,
        image_size,
        cache_size // 4,
        True,
        validate_files
    )

    print(f"{len(train_files)} files in training dataset")
    print(f"{len(validate_files)} files in validation dataset")
    return train_dataset, validate_dataset

class FaceDeblurDataset(Dataset):
    def __init__(self, blurred_path, origin_path, image_size=(240, 320), cache_size=1500, use_cache=True, file_list=None):
        self.blurred_path = blurred_path
        self.origin_path = origin_path
        self.image_size = image_size
        self.cache_size = cache_size
        self.use_cache = use_cache

        if file_list is None:
            blurred_files = set(os.listdir(blurred_path))
            origin_files = set(os.listdir(origin_path))
            self.image_files = list(blurred_files.intersection(origin_files))
            self.image_files = [i for i in self.image_files if i.endswith('.jpg')]
        else:
            self.image_files = file_list

        self.cache = {} if use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"Dataset: {len(self.image_files)} images")
        print(f"Size: {self.image_size}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        if self.cache is not None:
            cache_key = f"{image_name}_{self.image_size}"
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            self.cache_misses += 1

        blurred_path = os.path.join(self.blurred_path, image_name)
        origin_path = os.path.join(self.origin_path, image_name)

        blurred_image = cv2.cvtColor(cv2.imread(blurred_path), cv2.COLOR_BGR2RGB)
        origin_image = cv2.cvtColor(cv2.imread(origin_path), cv2.COLOR_BGR2RGB)

        blurred_tensor = torch.from_numpy(blurred_image.astype(np.float32)/255.0).permute(2, 0, 1)
        origin_tensor = torch.from_numpy(origin_image.astype(np.float32)/255.0).permute(2, 0, 1)

        # Horizontal flip augmentation
        if np.random.random() > 0.5:
            blurred_tensor = torch.flip(blurred_tensor, [2])
            origin_tensor = torch.flip(origin_tensor, [2])

        result = (blurred_tensor, origin_tensor, image_name)

        if self.cache is not None and len(self.cache) < self.cache_size:
            self.cache[cache_key] = result

        return result

    def get_cache_stats(self):
        if self.cache is None:
            return "Cache disabled"
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return f"Cache hit rate: {round(hit_rate, 2)} ({self.cache_hits}/{total})"

class ExperienceReplay:
    def __init__(self, max_size=3000, min_size=500):
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = deque(maxlen=max_size)
        self.current_size = 0

    def __len__(self):
        return len(self.buffer)

    def add_batch(self, blurred_batch, origin_batch, filenames_batch):
        batch_size = blurred_batch.size(0)

        for i in range(batch_size):
            experience = {
                'blurred': blurred_batch[i].clone().cpu(),
                'origin': origin_batch[i].clone().cpu(),
                'filename': filenames_batch[i]
            }
            self.buffer.append(experience)

        self.current_size = len(self.buffer)

    def sample_batch(self, batch_size, device='cuda'):
        if self.current_size < self.min_size:
            return None, None, None

        sample_size = min(batch_size, self.current_size)
        sampled_experiences = random.sample(list(self.buffer), sample_size)

        blurred_batch = torch.stack([exp['blurred'] for exp in sampled_experiences]).to(device)
        origin_batch = torch.stack([exp['origin'] for exp in sampled_experiences]).to(device)
        filenames_batch = [exp['filename'] for exp in sampled_experiences]

        return blurred_batch, origin_batch, filenames_batch

    def is_ready(self):
        return self.current_size >= self.min_size

    def get_stats(self):
        return {
            'buffer_size': self.current_size,
            'max_size': self.max_size,
            'is_ready': self.is_ready()
        }

class SeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = SeparableConvolution(in_channels, out_channels, stride=stride)
        self.conv2 = SeparableConvolution(out_channels, out_channels, stride=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)

class DeblurModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=32):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))

        self.enc1 = ResidualBlock(base_channels, base_channels, stride=1)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 3, stride=2)

        self.bottleneck = ResidualBlock(base_channels * 3, base_channels * 3)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 3, base_channels * 2, 3, 2, 1, 1), nn.BatchNorm2d(base_channels * 2), nn.ReLU(inplace=True))

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, 2, 1, 1), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))

        self.dec1 = nn.Sequential(nn.Conv2d(base_channels, base_channels // 2, 3, 1, 1), nn.BatchNorm2d(base_channels // 2), nn.ReLU(inplace=True))

        self.output_conv = nn.Conv2d(base_channels // 2, output_channels, 3, 1, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.output_conv(x)
        return torch.sigmoid(x)

class TrainingMonitor:
    def __init__(self, patience=5, min_delta=1e-6, save_best=True, best_loss=float('inf')):
        self.patience = patience
        self.min_delta = min_delta
        self.save_best = save_best
        self.best_loss = best_loss
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.validate_losses = []

    def update(self, train_loss, validate_loss, lr, epoch, model=None):
        self.train_losses.append(train_loss)
        self.validate_losses.append(validate_loss)

        improved = validate_loss < (self.best_loss - self.min_delta)

        if improved:
            self.best_loss = validate_loss
            self.epochs_without_improvement = 0
            if self.save_best and model is not None:
                torch.save(model.state_dict(), 'best_model.pth')
            return False, True
        else:
            self.epochs_without_improvement += 1
            early_stop = self.epochs_without_improvement >= self.patience
            return early_stop, False

    def is_converged(self, window=5):
        if len(self.validate_losses) < window * 2:
            return False

        recent_losses = self.validate_losses[-window:]
        older_losses = self.validate_losses[-window * 2:-window]
        recent_avg = sum(recent_losses) / len(recent_losses)
        older_avg = sum(older_losses) / len(older_losses)
        improvement = abs(older_avg - recent_avg) / older_avg
        return improvement < 0.001

def load_checkpoint(model, checkpoint_path, optimizer=None,
          scheduler=None, device='cuda'):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")
    else:
        model.load_state_dict(checkpoint)
        print("Model state loaded successfully (simple format)")

    loaded_info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_validate_loss': checkpoint.get('best_validate_loss', float('inf')),
        'train_losses': checkpoint.get('train_losses', []),
        'validate_losses': checkpoint.get('validate_losses', [])
    }

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded successfully")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded successfully")

    print(f"Checkpoint info: Epoch {loaded_info['epoch']}")
    print(f"Best validation loss: {round(loaded_info['best_validate_loss'],6)}")
    return loaded_info


def train(model, train_loader, validate_loader, checkpoint_path=None, num_epochs=50, learning_rate=0.001, early_stopping_patience=5, experience_replay_size=None, replay_dataset=None):
    device = torch.device('cuda')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    experience_replay = None
    replay_batches_used = 0
    total_start = time.time()

    # Load checkpoint if continue learning
    if checkpoint_path is not None:
        checkpoint_info = load_checkpoint(model, checkpoint_path, optimizer, scheduler, device)
        train_losses = checkpoint_info['train_losses'].copy()
        validate_losses = checkpoint_info['validate_losses'].copy()
        start_epoch = checkpoint_info['epoch']
        best_validate_loss = checkpoint_info['best_validate_loss']

        # Initialize experience replay to continue training
        if experience_replay_size is not None and replay_dataset is not None:
            experience_replay = ExperienceReplay(max_size=experience_replay_size, min_size=experience_replay_size // 6)

            replay_loader = DataLoader(replay_dataset, batch_size=32, shuffle=True)
            with torch.no_grad():
                for blurred, origin, filenames in replay_loader:
                    blurred = blurred.to(device, non_blocking=True)
                    origin = origin.to(device, non_blocking=True)
                    experience_replay.add_batch(blurred, origin, filenames)
                    if len(experience_replay) >= experience_replay_size:
                        break
    else:
        # Trening from scratch
        train_losses = []
        validate_losses = []
        start_epoch = 0
        best_validate_loss = float('inf')

    scaler = GradScaler()
    monitor = TrainingMonitor(patience=early_stopping_patience, save_best=True, best_loss=best_validate_loss)

    print(f"Starting from epoch: {start_epoch}")
    print(f"Epochs to run: {num_epochs}")

    for epoch in range(num_epochs):
        current_epoch = start_epoch + epoch + 1
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (blurred, origin, filenames) in enumerate(train_loader):
            blurred = blurred.to(device, non_blocking=True)
            origin = origin.to(device, non_blocking=True)

            train_blurred = blurred
            train_origin = origin

            # Mix with replay data (only 30%)
            if experience_replay is not None and experience_replay.is_ready() and random.random() < 0.3:
                replay_blurred,replay_origin,_ = experience_replay.sample_batch(batch_size=max(1, blurred.size(0) // 3), device=device)

                if replay_blurred is not None:
                    # Keep most of current batch and add some replay
                    current_size = blurred.size(0) * 2 // 3
                    train_blurred = torch.cat([blurred[:current_size], replay_blurred], dim=0)
                    train_origin = torch.cat([origin[:current_size], replay_origin], dim=0)
                    replay_batches_used = replay_batches_used + 1

            optimizer.zero_grad()

            with autocast():
                output = model(train_blurred)
                loss = F.mse_loss(output, train_origin)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            train_loss = train_loss + loss.item()

        # Validation
        model.eval()
        validate_loss = 0.0

        with torch.no_grad():
            for blurred, origin, filenames in validate_loader:
                blurred = blurred.to(device, non_blocking=True)
                origin = origin.to(device, non_blocking=True)

                with autocast():
                    output = model(blurred)
                    validate_loss = validate_loss + F.mse_loss(output, origin).item()

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_validate_loss = validate_loss / len(validate_loader)
        epoch_time = time.time() - epoch_start

        train_losses.append(avg_train_loss)
        validate_losses.append(avg_validate_loss)

        best_marker = ""
        early_stop, is_best = monitor.update(avg_train_loss, avg_validate_loss, scheduler.get_last_lr()[0], epoch, model)

        if early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
        if is_best:
            best_marker = " [BEST]"

        print(f"Epoch {current_epoch:3d})")
        print(f"Train={round(avg_train_loss,6)}, Val={round(avg_validate_loss,6)}, LR={round(scheduler.get_last_lr(),6)}, Time={round(epoch_time,1)}s{best_marker}")

        # Save checkpoint every 5 epochs
        if current_epoch % 5 == 0:
            checkpoint_filename = f'model_epoch_{current_epoch}.pth'
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'validate_loss': avg_validate_loss,
                'train_losses': train_losses,
                'validate_losses': validate_losses,
                'best_validate_loss': monitor.best_loss
            }, checkpoint_filename)
            print(f"Checkpoint saved: {checkpoint_filename}")

        if current_epoch % 10 == 0:
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    final_best_loss = monitor.best_loss
    end_time = time.strftime("%H:%M:%S")

    print(f"Total training time: {round(total_time/3600,2)}h {end_time}")
    print(f"Best validation loss: {round(final_best_loss,6)}")
    print(f"Replay batches used: {replay_batches_used}")

    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_epoch': len(train_losses),
        'best_validate_loss': final_best_loss,
        'train_losses': train_losses,
        'validate_losses': validate_losses,
        'training_time_hours': total_time / 3600,
        'end_time': end_time
    }
    torch.save(final_checkpoint, 'final_model.pth')
    print(f"Final model saved: final_model.pth")

    return model, train_losses, validate_losses

if __name__ == "__main__":
    origin_path = input("Enter copied path to original images: ")
    blurred_path = input("Enter copied path to blurred images: ")

    # Initialize deblur model
    model = DeblurModel(base_channels=32)

    # Split dataset to training and validation
    train_dataset, validate_dataset = create_datasets(
        blurred_path=blurred_path,
        origin_path=origin_path,
        image_size=(240, 320),
        cache_size=1000,
        random_seed=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    validate_loader = DataLoader(
        validate_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    train_option = input("Do you want to continue learning? y/n\n")

    if train_option in ["y", "Y", "yes"]:
        # Continue training from checkpoint
        old_origin_path = input("Enter path to original images: ")
        old_blurred_path = input("Enter path to blurred images: ")
        checkpoint_path = input("Enter copied path with trained: ")

        replay_dataset = FaceDeblurDataset(
            old_blurred_path,
            old_origin_path,
            (240, 320),
            500,
            True
        )

        model, train_losses, validate_losses = train(
            model=model,
            train_loader=train_loader,
            validate_loader=validate_loader,
            checkpoint_path=checkpoint_path,
            num_epochs=30,
            learning_rate=0.0003,
            early_stopping_patience=5,
            experience_replay_size=5000,
            replay_dataset=replay_dataset
        )
    else:
        # Start training from scratch
        model, train_losses, validate_losses = train(
            model=model,
            train_loader=train_loader,
            validate_loader=validate_loader,
            checkpoint_path=None,
            num_epochs=50,
            learning_rate=0.001,
            early_stopping_patience=5
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()