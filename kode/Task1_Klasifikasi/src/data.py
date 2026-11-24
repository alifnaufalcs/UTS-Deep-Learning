"""
Optimized In-Memory Vectorized Dataset and DataLoader for Task 1
Sub-second iteration speed for 48,000 samples by eliminating per-item Python overhead.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

class FastMultimodalDataset(Dataset):
    def __init__(self, images_np, sensors_np, labels_np=None, meta_df=None, norm_stats=None, is_train=False):
        """
        Pre-vectorizes all arrays into contiguous PyTorch tensors.
        Images are stored in uint8 (saving 75% memory), normalized on GPU in batch.
        Sensors and masks are pre-normalized and stored in float32.
        """
        N = len(images_np)
        self.N = N
        self.is_train = is_train
        self.meta_df = meta_df
        
        # 1. Images: (N, 64, 64) -> (N, 1, 64, 64) uint8 tensor
        if images_np.ndim == 3:
            images_np = np.expand_dims(images_np, axis=1)
        self.images = torch.from_numpy(np.ascontiguousarray(images_np))
        
        # 2. Sensor Pre-normalization and Masking
        sensor_is_nan = np.isnan(sensors_np).all(axis=(1, 2))
        sensor_mask_np = (~sensor_is_nan).astype(np.float32)[:, np.newaxis] # (N, 1)
        
        if norm_stats is not None:
            self.img_mean = float(norm_stats['img_mean'])
            self.img_std = float(norm_stats['img_std'])
            s_means = np.array(norm_stats['sensor_means'], dtype=np.float32).reshape(1, 3, 1)
            s_stds = np.array(norm_stats['sensor_stds'], dtype=np.float32).reshape(1, 3, 1)
        else:
            self.img_mean = 0.5
            self.img_std = 0.5
            s_means = np.zeros((1, 3, 1), dtype=np.float32)
            s_stds = np.ones((1, 3, 1), dtype=np.float32)
            
        # Impute NaNs with 0.0, normalize, and zero out missing ones
        sensors_clean = np.nan_to_num(sensors_np, nan=0.0).astype(np.float32)
        sensors_clean = (sensors_clean - s_means) / s_stds
        sensors_clean[sensor_is_nan] = 0.0
        
        self.sensors = torch.from_numpy(np.ascontiguousarray(sensors_clean)) # (N, 3, 128)
        self.masks = torch.from_numpy(np.ascontiguousarray(sensor_mask_np)) # (N, 1)
        
        # 3. Labels
        if labels_np is not None:
            self.labels = torch.from_numpy(np.ascontiguousarray(labels_np, dtype=np.int64))
        else:
            self.labels = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        item = {
            'image': self.images[idx],
            'sensor': self.sensors[idx],
            'sensor_mask': self.masks[idx],
            'index': idx
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item

def get_dataloaders(cache_dir='C:/DL_Classification/data/cache',
                    batch_size=64,
                    val_ratio=0.2,
                    seed=42,
                    p_modality_drop=0.15,
                    num_workers=0,
                    quick_subset=None):
    """
    Creates stratified train and validation DataLoaders.
    Fits normalization stats strictly on train split.
    """
    train_images = np.load(os.path.join(cache_dir, 'train_images.npy'))
    train_sensors = np.load(os.path.join(cache_dir, 'train_sensors.npy'))
    train_labels = np.load(os.path.join(cache_dir, 'train_labels.npy'))
    train_meta = pd.read_parquet(os.path.join(cache_dir, 'train_meta.parquet'))
    
    if quick_subset is not None and quick_subset < len(train_images):
        print(f"[Quick Mode] Subsetting data to {quick_subset} samples...")
        indices = np.random.RandomState(seed).permutation(len(train_images))[:quick_subset]
        train_images = train_images[indices]
        train_sensors = train_sensors[indices]
        train_labels = train_labels[indices]
        train_meta = train_meta.iloc[indices].reset_index(drop=True)

    stratify_col = train_meta['kelas'] + '_' + train_meta['lini']
    skf = StratifiedKFold(n_splits=int(1/val_ratio), shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(train_images, stratify_col))
    
    print(f"Data Split: Train={len(train_idx)} samples, Validation={len(val_idx)} samples")
    
    from .utils import compute_train_norm_stats
    norm_stats = compute_train_norm_stats(
        train_images[train_idx],
        train_sensors[train_idx],
        save_path=os.path.join(cache_dir, 'norm_stats.json')
    )
    
    train_dataset = FastMultimodalDataset(
        images_np=train_images[train_idx],
        sensors_np=train_sensors[train_idx],
        labels_np=train_labels[train_idx],
        meta_df=train_meta.iloc[train_idx].reset_index(drop=True),
        norm_stats=norm_stats,
        is_train=True
    )
    
    val_dataset = FastMultimodalDataset(
        images_np=train_images[val_idx],
        sensors_np=train_sensors[val_idx],
        labels_np=train_labels[val_idx],
        meta_df=train_meta.iloc[val_idx].reset_index(drop=True),
        norm_stats=norm_stats,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, norm_stats, (train_idx, val_idx)

def get_test_dataloader(cache_dir='C:/DL_Classification/data/cache',
                        batch_size=64,
                        norm_stats=None,
                        num_workers=0):
    test_images = np.load(os.path.join(cache_dir, 'test_images.npy'))
    test_sensors = np.load(os.path.join(cache_dir, 'test_sensors.npy'))
    test_meta = pd.read_parquet(os.path.join(cache_dir, 'test_meta.parquet'))
    
    if norm_stats is None:
        with open(os.path.join(cache_dir, 'norm_stats.json')) as f:
            norm_stats = json.load(f)
            
    test_dataset = FastMultimodalDataset(
        images_np=test_images,
        sensors_np=test_sensors,
        labels_np=None,
        meta_df=test_meta,
        norm_stats=norm_stats,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    return test_loader, test_meta
