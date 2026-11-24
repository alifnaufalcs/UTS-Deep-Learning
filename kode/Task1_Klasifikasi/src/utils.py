"""
Utility functions for Task 1: Multimodal Defect Classification
Reproducibility, metrics calculation, plotting, and normalization helpers.
"""

import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

def seed_everything(seed=42):
    """Ensure full reproducibility across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def calculate_metrics(y_true, y_pred, class_names):
    """
    Compute Macro-F1 (competition primary metric), weighted F1, per-class F1,
    accuracy, and confusion matrix.
    """
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    per_class_dict = {f"f1_{class_names[i]}": float(per_class_f1[i]) for i in range(len(class_names))}
    
    return {
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(acc),
        'per_class_f1': per_class_dict,
        'confusion_matrix': cm.tolist()
    }

def plot_confusion_matrix(cm, class_names, title, save_path):
    """Plot and save a clean, normalized confusion matrix."""
    cm = np.array(cm)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Persentase (%)'}, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel('Label Sebenarnya (Ground Truth)', fontsize=11)
    ax.set_xlabel('Label Prediksi (Predicted)', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def get_class_weights(labels, num_classes=7, method='inverse_freq'):
    """
    Calculate class weights to mitigate class imbalance.
    Weights are inversely proportional to class frequencies.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    if method == 'inverse_freq':
        weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    elif method == 'sqrt_inverse':
        weights = np.sqrt(total / (num_classes * np.maximum(counts, 1).astype(np.float32)))
        weights = weights / weights.mean()
    elif method == 'effective_num':
        beta = 0.999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-6)
        weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)

def compute_train_norm_stats(train_images, train_sensors, save_path=None):
    """
    Compute normalization statistics solely from train split (prevent data leakage).
    """
    img_mean = float(train_images.mean() / 255.0)
    img_std = float(train_images.std() / 255.0)
    
    sensor_means = []
    sensor_stds = []
    for c in range(train_sensors.shape[1]):
        ch_vals = train_sensors[:, c, :].ravel()
        valid = ch_vals[~np.isnan(ch_vals)]
        sensor_means.append(float(valid.mean()))
        sensor_stds.append(float(valid.std() + 1e-6))
        
    stats = {
        'img_mean': img_mean,
        'img_std': img_std,
        'sensor_means': sensor_means,
        'sensor_stds': sensor_stds
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
    return stats
