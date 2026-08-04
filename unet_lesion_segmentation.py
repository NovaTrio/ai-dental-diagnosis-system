# """
# unet_lesion_segmentation.py
# ────────────────────────────
# Periapical lesion segmentation using U-Net architecture.

# Pipeline:
#   1. Load texture_removed images (from texture_output/) and their
#      corresponding binary masks (from mask/).
#   2. Build a U-Net model in PyTorch.
#   3. Train with Dice + BCE combined loss.
#   4. Evaluate with Dice coefficient, IoU, Precision, and Recall.
#   5. Run inference and save predicted masks + overlays.

# Input:
#   - Images : data/abscess/raw/texture_output/texture_removed_*.png
#   - Masks  : data/abscess/raw/mask/texture_removed_*_mask_*.png
#              (white = lesion, black = background)

# Output:
#   - Trained model weights saved to data/abscess/raw/unet_output/
#   - Predicted masks + overlays saved to data/abscess/raw/unet_output/predictions/
#   - Training metrics + loss curves saved to data/abscess/raw/unet_output/

# Usage:
#   python unet_lesion_segmentation.py
# """

# # ─────────────────────────────────────────────────────────────────────────────
# # Step 1: Import Libraries
# # ─────────────────────────────────────────────────────────────────────────────
# import os
# import re
# import json
# import random
# import numpy as np
# import cv2
# from glob import glob
# from datetime import datetime

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms.functional as TF

# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
# import matplotlib.pyplot as plt

# # ─────────────────────────────────────────────────────────────────────────────
# # Configuration
# # ─────────────────────────────────────────────────────────────────────────────
# # Paths (relative to this script's location in src/abscess/preprocessing/)
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# IMAGE_DIR  = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', 'abscess', 'raw', 'texture_output')
# MASK_DIR   = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', 'abscess', 'raw', 'mask')
# OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', 'data', 'abscess', 'raw', 'unet_output')

# # Training hyperparameters
# IMG_SIZE       = 256       # Resize images to IMG_SIZE x IMG_SIZE
# BATCH_SIZE     = 4
# LEARNING_RATE  = 1e-4
# NUM_EPOCHS     = 100
# VAL_SPLIT      = 0.2      # 20% for validation
# RANDOM_SEED    = 42

# # Model settings
# IN_CHANNELS    = 1         # Grayscale input
# OUT_CHANNELS   = 1         # Binary segmentation (lesion / background)
# FEATURES       = [64, 128, 256, 512]  # U-Net encoder channel sizes

# # Device
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 2: Dataset — Pair texture_removed images with their masks
# # ─────────────────────────────────────────────────────────────────────────────
# def build_image_mask_pairs(image_dir: str, mask_dir: str):
#     """
#     Match each texture_removed image to its corresponding mask.

#     Image naming:  texture_removed_L{N}_clahe_sigmoid.png
#     Mask naming :  texture_removed_L{N}_clahe_sigmoid_mask_1_L{N}.png
#                    (some have slightly different suffixes)

#     Strategy: For each image, extract "L{N}" and find the mask file that
#     contains the same "L{N}" identifier.
#     """
#     supported_exts = ('.png', '.jpg', '.jpeg')

#     # Collect all texture_removed images
#     image_files = sorted([
#         f for f in os.listdir(image_dir)
#         if f.lower().startswith('texture_removed_') and f.lower().endswith(supported_exts)
#     ])

#     # Collect all mask files
#     mask_files = sorted([
#         f for f in os.listdir(mask_dir)
#         if f.lower().endswith(supported_exts)
#     ])

#     pairs = []

#     for img_file in image_files:
#         # Extract the L-number identifier, e.g. "L1", "L10", etc.
#         match = re.search(r'(L\d+)', img_file)
#         if not match:
#             print(f"  [WARN] Could not extract L-number from: {img_file}")
#             continue

#         l_id = match.group(1)  # e.g., "L1"

#         # Find a matching mask file that contains this L-id
#         # The mask files have format: texture_removed_L{N}_clahe_sigmoid_mask_1_L{N}.png
#         matching_masks = [
#             m for m in mask_files
#             if m.startswith(f'texture_removed_{l_id}_clahe_sigmoid_mask')
#         ]

#         if matching_masks:
#             pairs.append((
#                 os.path.join(image_dir, img_file),
#                 os.path.join(mask_dir, matching_masks[0])
#             ))
#         else:
#             print(f"  [WARN] No mask found for image: {img_file} (L-id: {l_id})")

#     return pairs


# class LesionDataset(Dataset):
#     """
#     Custom Dataset for lesion segmentation.
#     Loads image-mask pairs, resizes them, and applies data augmentation.
#     """

#     def __init__(self, image_mask_pairs, img_size=256, augment=False):
#         self.pairs = image_mask_pairs
#         self.img_size = img_size
#         self.augment = augment

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         img_path, mask_path = self.pairs[idx]

#         # ── Load image ──
#         image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         if image is None:
#             raise FileNotFoundError(f"Could not read image: {img_path}")

#         # Convert to grayscale
#         if len(image.shape) == 3:
#             if image.shape[2] == 4:   # RGBA
#                 image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
#             else:                      # BGR
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # ── Load mask ──
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         if mask is None:
#             raise FileNotFoundError(f"Could not read mask: {mask_path}")

#         # Binarize mask: white (>=128) → 1, black (<128) → 0
#         _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

#         # ── Resize ──
#         image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
#         mask  = cv2.resize(mask,  (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

#         # ── Data Augmentation ──
#         if self.augment:
#             image, mask = self._augment(image, mask)

#         # ── Normalize ──
#         image = image.astype(np.float32) / 255.0  # [0, 1]
#         mask  = mask.astype(np.float32) / 255.0   # {0, 1}

#         # ── Convert to tensors: (C, H, W) ──
#         image = torch.from_numpy(image).unsqueeze(0)   # (1, H, W)
#         mask  = torch.from_numpy(mask).unsqueeze(0)     # (1, H, W)

#         return image, mask

#     def _augment(self, image, mask):
#         """Apply random augmentations (same transform to both image and mask)."""

#         # Random horizontal flip
#         if random.random() > 0.5:
#             image = cv2.flip(image, 1)
#             mask  = cv2.flip(mask, 1)

#         # Random vertical flip
#         if random.random() > 0.5:
#             image = cv2.flip(image, 0)
#             mask  = cv2.flip(mask, 0)

#         # Random rotation (0°, 90°, 180°, 270°)
#         k = random.randint(0, 3)
#         if k > 0:
#             image = np.rot90(image, k).copy()
#             mask  = np.rot90(mask, k).copy()

#         # Random brightness/contrast adjustment (image only)
#         if random.random() > 0.5:
#             alpha = random.uniform(0.8, 1.2)  # contrast
#             beta  = random.randint(-20, 20)    # brightness
#             image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

#         # Random Gaussian noise (image only)
#         if random.random() > 0.5:
#             noise = np.random.normal(0, 5, image.shape).astype(np.float32)
#             image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

#         return image, mask


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 3: U-Net Architecture
# # ─────────────────────────────────────────────────────────────────────────────
# class DoubleConv(nn.Module):
#     """Two consecutive (Conv2d → BatchNorm → ReLU) blocks."""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class UNet(nn.Module):
#     """
#     U-Net architecture for binary segmentation.

#     Architecture:
#         Encoder: 4 downsampling blocks (DoubleConv + MaxPool)
#         Bottleneck: DoubleConv at the lowest resolution
#         Decoder: 4 upsampling blocks (ConvTranspose2d + skip connection + DoubleConv)
#         Output: 1×1 Conv → Sigmoid
#     """

#     def __init__(self, in_channels=1, out_channels=1, features=None):
#         super().__init__()
#         if features is None:
#             features = [64, 128, 256, 512]

#         self.encoder_blocks = nn.ModuleList()
#         self.decoder_blocks = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.upconvs = nn.ModuleList()

#         # ── Encoder ──
#         prev_channels = in_channels
#         for feature in features:
#             self.encoder_blocks.append(DoubleConv(prev_channels, feature))
#             prev_channels = feature

#         # ── Bottleneck ──
#         self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

#         # ── Decoder ──
#         for feature in reversed(features):
#             self.upconvs.append(
#                 nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
#             )
#             self.decoder_blocks.append(DoubleConv(feature * 2, feature))

#         # ── Final 1×1 convolution ──
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#     def forward(self, x):
#         skip_connections = []

#         # ── Encoder path ──
#         for encoder in self.encoder_blocks:
#             x = encoder(x)
#             skip_connections.append(x)
#             x = self.pool(x)

#         # ── Bottleneck ──
#         x = self.bottleneck(x)

#         # ── Decoder path ──
#         skip_connections = skip_connections[::-1]  # Reverse for decoder order

#         for idx in range(len(self.decoder_blocks)):
#             x = self.upconvs[idx](x)

#             skip = skip_connections[idx]

#             # Handle size mismatch (if input size is not perfectly divisible)
#             if x.shape != skip.shape:
#                 x = TF.resize(x, size=skip.shape[2:])

#             # Concatenate skip connection
#             x = torch.cat([skip, x], dim=1)
#             x = self.decoder_blocks[idx](x)

#         return torch.sigmoid(self.final_conv(x))


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 4: Loss Function — Dice + BCE Combined
# # ─────────────────────────────────────────────────────────────────────────────
# class DiceBCELoss(nn.Module):
#     """
#     Combined Dice Loss + Binary Cross-Entropy Loss.
#     Dice Loss handles class imbalance (small lesion regions).
#     BCE Loss provides stable gradient flow.
#     """

#     def __init__(self, smooth=1.0):
#         super().__init__()
#         self.smooth = smooth
#         self.bce = nn.BCELoss()

#     def forward(self, predictions, targets):
#         # BCE component
#         bce_loss = self.bce(predictions, targets)

#         # Dice component
#         predictions_flat = predictions.view(-1)
#         targets_flat = targets.view(-1)

#         intersection = (predictions_flat * targets_flat).sum()
#         dice_coeff = (2.0 * intersection + self.smooth) / (
#             predictions_flat.sum() + targets_flat.sum() + self.smooth
#         )
#         dice_loss = 1.0 - dice_coeff

#         return bce_loss + dice_loss


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 5: Evaluation Metrics
# # ─────────────────────────────────────────────────────────────────────────────
# def compute_metrics(predictions, targets, threshold=0.5):
#     """
#     Compute Dice coefficient, IoU, Precision, and Recall.

#     Args:
#         predictions: Model output (probabilities), shape (B, 1, H, W)
#         targets:     Ground truth masks, shape (B, 1, H, W)
#         threshold:   Binarization threshold

#     Returns:
#         Dictionary with dice, iou, precision, recall values.
#     """
#     preds_binary = (predictions > threshold).float()
#     targets_binary = targets.float()

#     # Flatten
#     preds_flat = preds_binary.view(-1)
#     targets_flat = targets_binary.view(-1)

#     # True Positives, False Positives, False Negatives
#     tp = (preds_flat * targets_flat).sum().item()
#     fp = (preds_flat * (1 - targets_flat)).sum().item()
#     fn = ((1 - preds_flat) * targets_flat).sum().item()

#     smooth = 1e-7

#     dice      = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
#     iou       = (tp + smooth) / (tp + fp + fn + smooth)
#     precision = (tp + smooth) / (tp + fp + smooth)
#     recall    = (tp + smooth) / (tp + fn + smooth)

#     return {
#         'dice': dice,
#         'iou': iou,
#         'precision': precision,
#         'recall': recall,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 6: Training Loop
# # ─────────────────────────────────────────────────────────────────────────────
# def train_one_epoch(model, dataloader, criterion, optimizer, device):
#     """Train for one epoch. Returns average loss and metrics."""
#     model.train()
#     running_loss = 0.0
#     all_metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
#     num_batches = 0

#     for images, masks in dataloader:
#         images = images.to(device)
#         masks  = masks.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, masks)

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Accumulate loss and metrics
#         running_loss += loss.item()
#         batch_metrics = compute_metrics(outputs.detach(), masks.detach())
#         for key in all_metrics:
#             all_metrics[key] += batch_metrics[key]
#         num_batches += 1

#     # Average
#     avg_loss = running_loss / max(num_batches, 1)
#     for key in all_metrics:
#         all_metrics[key] /= max(num_batches, 1)

#     return avg_loss, all_metrics


# def validate(model, dataloader, criterion, device):
#     """Validate the model. Returns average loss and metrics."""
#     model.eval()
#     running_loss = 0.0
#     all_metrics = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
#     num_batches = 0

#     with torch.no_grad():
#         for images, masks in dataloader:
#             images = images.to(device)
#             masks  = masks.to(device)

#             outputs = model(images)
#             loss = criterion(outputs, masks)

#             running_loss += loss.item()
#             batch_metrics = compute_metrics(outputs, masks)
#             for key in all_metrics:
#                 all_metrics[key] += batch_metrics[key]
#             num_batches += 1

#     avg_loss = running_loss / max(num_batches, 1)
#     for key in all_metrics:
#         all_metrics[key] /= max(num_batches, 1)

#     return avg_loss, all_metrics


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 7: Inference & Visualization
# # ─────────────────────────────────────────────────────────────────────────────
# def run_inference(model, dataset, device, output_dir):
#     """
#     Run inference on the full dataset and save:
#       1. Predicted binary mask
#       2. Overlay of prediction on original image
#       3. Side-by-side comparison (image | ground truth | prediction)
#     """
#     pred_dir = os.path.join(output_dir, 'predictions')
#     os.makedirs(pred_dir, exist_ok=True)

#     model.eval()
#     with torch.no_grad():
#         for idx in range(len(dataset)):
#             image, gt_mask = dataset[idx]
#             image_batch = image.unsqueeze(0).to(device)

#             # Predict
#             pred = model(image_batch).squeeze().cpu().numpy()
#             pred_binary = (pred > 0.5).astype(np.uint8) * 255

#             # Convert tensors to numpy for visualization
#             img_np = (image.squeeze().numpy() * 255).astype(np.uint8)
#             gt_np  = (gt_mask.squeeze().numpy() * 255).astype(np.uint8)

#             # Get original filename from dataset
#             img_path = dataset.pairs[idx][0]
#             base_name = os.path.splitext(os.path.basename(img_path))[0]

#             # ── Save predicted mask ──
#             cv2.imwrite(
#                 os.path.join(pred_dir, f'pred_mask_{base_name}.png'),
#                 pred_binary
#             )

#             # ── Save overlay ──
#             overlay = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
#             # Highlight predicted lesion in red
#             overlay[pred_binary == 255] = [0, 0, 255]  # BGR → red
#             cv2.imwrite(
#                 os.path.join(pred_dir, f'pred_overlay_{base_name}.png'),
#                 overlay
#             )

#             # ── Save comparison figure ──
#             fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#             axes[0].imshow(img_np, cmap='gray')
#             axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
#             axes[0].axis('off')

#             axes[1].imshow(gt_np, cmap='gray')
#             axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
#             axes[1].axis('off')

#             axes[2].imshow(pred_binary, cmap='gray')
#             axes[2].set_title('U-Net Prediction', fontsize=12, fontweight='bold')
#             axes[2].axis('off')

#             plt.suptitle(base_name, fontsize=14, fontweight='bold')
#             plt.tight_layout()
#             plt.savefig(
#                 os.path.join(pred_dir, f'comparison_{base_name}.png'),
#                 dpi=150, bbox_inches='tight'
#             )
#             plt.close(fig)

#     print(f"  → Predictions saved to: {pred_dir}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Step 8: Plot Training Curves
# # ─────────────────────────────────────────────────────────────────────────────
# def plot_training_curves(history, output_dir):
#     """Plot and save training/validation loss and Dice curves."""

#     epochs = range(1, len(history['train_loss']) + 1)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # ── Loss curves ──
#     axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
#     axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
#     axes[0].set_xlabel('Epoch', fontsize=12)
#     axes[0].set_ylabel('Loss (Dice + BCE)', fontsize=12)
#     axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
#     axes[0].legend(fontsize=11)
#     axes[0].grid(True, alpha=0.3)

#     # ── Dice coefficient curves ──
#     axes[1].plot(epochs, history['train_dice'], 'b-', linewidth=2, label='Train Dice')
#     axes[1].plot(epochs, history['val_dice'], 'r-', linewidth=2, label='Val Dice')
#     axes[1].set_xlabel('Epoch', fontsize=12)
#     axes[1].set_ylabel('Dice Coefficient', fontsize=12)
#     axes[1].set_title('Training & Validation Dice', fontsize=14, fontweight='bold')
#     axes[1].legend(fontsize=11)
#     axes[1].grid(True, alpha=0.3)
#     axes[1].set_ylim([0, 1])

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  → Training curves saved to: {os.path.join(output_dir, 'training_curves.png')}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     print("=" * 70)
#     print("  U-Net Lesion Segmentation Pipeline")
#     print("=" * 70)
#     print(f"  Device         : {DEVICE}")
#     print(f"  Image size     : {IMG_SIZE} × {IMG_SIZE}")
#     print(f"  Batch size     : {BATCH_SIZE}")
#     print(f"  Learning rate  : {LEARNING_RATE}")
#     print(f"  Epochs         : {NUM_EPOCHS}")
#     print(f"  Val split      : {VAL_SPLIT}")
#     print("=" * 70)

#     # Set random seeds for reproducibility
#     random.seed(RANDOM_SEED)
#     np.random.seed(RANDOM_SEED)
#     torch.manual_seed(RANDOM_SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(RANDOM_SEED)

#     # Create output directory
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # ── Step A: Build image-mask pairs ──
#     print("\n[1/6] Building image-mask pairs...")
#     pairs = build_image_mask_pairs(IMAGE_DIR, MASK_DIR)
#     print(f"  Found {len(pairs)} valid image-mask pairs.")

#     if len(pairs) == 0:
#         print("\n[ERROR] No image-mask pairs found. Check your data directories.")
#         print(f"  Images dir : {IMAGE_DIR}")
#         print(f"  Masks dir  : {MASK_DIR}")
#         exit(1)

#     # ── Step B: Train/Validation split ──
#     print("\n[2/6] Splitting dataset...")
#     random.shuffle(pairs)
#     val_count = max(1, int(len(pairs) * VAL_SPLIT))
#     val_pairs   = pairs[:val_count]
#     train_pairs = pairs[val_count:]
#     print(f"  Training samples   : {len(train_pairs)}")
#     print(f"  Validation samples : {len(val_pairs)}")

#     # ── Step C: Create datasets & dataloaders ──
#     train_dataset = LesionDataset(train_pairs, img_size=IMG_SIZE, augment=True)
#     val_dataset   = LesionDataset(val_pairs,   img_size=IMG_SIZE, augment=False)

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
#     val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

#     # ── Step D: Initialize model, loss, optimizer, scheduler ──
#     print("\n[3/6] Initializing U-Net model...")
#     model = UNet(
#         in_channels=IN_CHANNELS,
#         out_channels=OUT_CHANNELS,
#         features=FEATURES
#     ).to(DEVICE)

#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  Total parameters     : {total_params:,}")
#     print(f"  Trainable parameters : {trainable_params:,}")

#     criterion = DiceBCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=10
#     )

#     # ── Step E: Training loop ──
#     print("\n[4/6] Starting training...")
#     print("-" * 70)

#     history = {
#         'train_loss': [], 'val_loss': [],
#         'train_dice': [], 'val_dice': [],
#         'train_iou': [],  'val_iou': [],
#     }

#     best_val_dice = 0.0
#     best_epoch = 0
#     patience_counter = 0
#     EARLY_STOP_PATIENCE = 25

#     for epoch in range(1, NUM_EPOCHS + 1):
#         # Train
#         train_loss, train_metrics = train_one_epoch(
#             model, train_loader, criterion, optimizer, DEVICE
#         )

#         # Validate
#         val_loss, val_metrics = validate(
#             model, val_loader, criterion, DEVICE
#         )

#         # Update scheduler
#         scheduler.step(val_loss)

#         # Record history
#         history['train_loss'].append(train_loss)
#         history['val_loss'].append(val_loss)
#         history['train_dice'].append(train_metrics['dice'])
#         history['val_dice'].append(val_metrics['dice'])
#         history['train_iou'].append(train_metrics['iou'])
#         history['val_iou'].append(val_metrics['iou'])

#         # Print progress
#         if epoch % 5 == 0 or epoch == 1:
#             print(
#                 f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
#                 f"Train Loss: {train_loss:.4f}  Dice: {train_metrics['dice']:.4f} | "
#                 f"Val Loss: {val_loss:.4f}  Dice: {val_metrics['dice']:.4f}  "
#                 f"IoU: {val_metrics['iou']:.4f}"
#             )

#         # Save best model
#         if val_metrics['dice'] > best_val_dice:
#             best_val_dice = val_metrics['dice']
#             best_epoch = epoch
#             patience_counter = 0
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_dice': best_val_dice,
#                 'val_loss': val_loss,
#             }, os.path.join(OUTPUT_DIR, 'best_unet_model.pth'))
#         else:
#             patience_counter += 1

#         # Early stopping
#         if patience_counter >= EARLY_STOP_PATIENCE:
#             print(f"\n  [Early Stopping] No improvement for {EARLY_STOP_PATIENCE} epochs. "
#                   f"Best Dice: {best_val_dice:.4f} at epoch {best_epoch}.")
#             break

#     print("-" * 70)
#     print(f"  Training complete!")
#     print(f"  Best validation Dice : {best_val_dice:.4f} (epoch {best_epoch})")

#     # ── Step F: Plot training curves ──
#     print("\n[5/6] Plotting training curves...")
#     plot_training_curves(history, OUTPUT_DIR)

#     # ── Step G: Load best model & run inference ──
#     print("\n[6/6] Running inference with best model...")
#     checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_unet_model.pth'), map_location=DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print(f"  Loaded best model from epoch {checkpoint['epoch']} "
#           f"(Val Dice: {checkpoint['val_dice']:.4f})")

#     # Build a full dataset (no augmentation) for inference
#     full_dataset = LesionDataset(pairs, img_size=IMG_SIZE, augment=False)
#     run_inference(model, full_dataset, DEVICE, OUTPUT_DIR)

#     # ── Final evaluation on full dataset ──
#     print("\n  Final evaluation on all data:")
#     full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#     _, final_metrics = validate(model, full_loader, criterion, DEVICE)
#     print(f"    Dice Coefficient : {final_metrics['dice']:.4f}")
#     print(f"    IoU (Jaccard)    : {final_metrics['iou']:.4f}")
#     print(f"    Precision        : {final_metrics['precision']:.4f}")
#     print(f"    Recall           : {final_metrics['recall']:.4f}")

#     # ── Save metrics to JSON ──
#     metrics_report = {
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         'device': str(DEVICE),
#         'hyperparameters': {
#             'img_size': IMG_SIZE,
#             'batch_size': BATCH_SIZE,
#             'learning_rate': LEARNING_RATE,
#             'num_epochs': NUM_EPOCHS,
#             'features': FEATURES,
#         },
#         'best_epoch': best_epoch,
#         'best_val_dice': best_val_dice,
#         'final_metrics': final_metrics,
#         'total_parameters': total_params,
#         'num_train_samples': len(train_pairs),
#         'num_val_samples': len(val_pairs),
#     }
#     with open(os.path.join(OUTPUT_DIR, 'training_report.json'), 'w') as f:
#         json.dump(metrics_report, f, indent=2)

#     print(f"\n  → Training report saved to: {os.path.join(OUTPUT_DIR, 'training_report.json')}")
#     print("\n" + "=" * 70)
#     print("  U-Net Lesion Segmentation — COMPLETE")
#     print("=" * 70)
