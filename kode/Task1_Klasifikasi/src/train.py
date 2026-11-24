"""
High-Performance Training Engine for Task 1: Multimodal Defect Classification
Vectorized GPU-accelerated batch normalization and augmentations.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .utils import calculate_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device,
                    img_mean=0.5, img_std=0.5, p_modality_drop=0.15):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    use_cuda = device.type == 'cuda'
    
    for batch in dataloader:
        raw_images = batch['image'].to(device, non_blocking=True)
        sensors = batch['sensor'].to(device, non_blocking=True)
        masks = batch['sensor_mask'].to(device, non_blocking=True)
        targets = batch['label'].to(device, non_blocking=True)
        B = raw_images.size(0)
        
        # Batch GPU normalization
        images = (raw_images.float() / 255.0 - img_mean) / img_std
        
        # Fast GPU Batch Augmentations
        if torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[-1]) # Horizontal flip
        if torch.rand(1).item() > 0.5:
            images = torch.flip(images, dims=[-2]) # Vertical flip
            
        # GPU Modality Dropout (Task f)
        if p_modality_drop > 0:
            drop = (torch.rand(B, 1, device=device) < p_modality_drop).float()
            masks = masks * (1.0 - drop)
            sensors = sensors * masks.unsqueeze(-1)
            
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(images, sensors, masks)
            loss = criterion(logits, targets)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * B
        all_preds.append(torch.argmax(logits, dim=1).detach())
        all_targets.append(targets.detach())
        
    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    return avg_loss, all_preds, all_targets

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, class_names,
                   img_mean=0.5, img_std=0.5):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    use_cuda = device.type == 'cuda'
    
    for batch in dataloader:
        raw_images = batch['image'].to(device, non_blocking=True)
        sensors = batch['sensor'].to(device, non_blocking=True)
        masks = batch['sensor_mask'].to(device, non_blocking=True)
        targets = batch['label'].to(device, non_blocking=True)
        B = raw_images.size(0)
        
        images = (raw_images.float() / 255.0 - img_mean) / img_std
        
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(images, sensors, masks)
            loss = criterion(logits, targets)
            
        total_loss += loss.item() * B
        all_probs.append(torch.softmax(logits, dim=1).detach())
        all_targets.append(targets.detach())
        
    avg_loss = total_loss / len(dataloader.dataset)
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_preds = np.argmax(all_probs, axis=1)
    
    metrics = calculate_metrics(all_targets, all_preds, class_names)
    metrics['loss'] = float(avg_loss)
    metrics['probs'] = all_probs
    return metrics

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler,
                device, class_names, num_epochs=15, patience=5,
                checkpoint_path='artifacts/task1/best_model.pt',
                verbose=True, p_modality_drop=0.15):
    """
    Full training loop with Early Stopping and learning curve tracking.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    scaler = torch.amp.GradScaler(device='cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))
    
    img_mean = train_loader.dataset.img_mean
    img_std = train_loader.dataset.img_std
    
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_macro_f1': [], 'val_macro_f1': [],
        'epoch_times': []
    }
    
    start_train_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        
        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            img_mean=img_mean, img_std=img_std, p_modality_drop=p_modality_drop
        )
        train_metrics = calculate_metrics(train_targets, train_preds, class_names)
        
        val_metrics = evaluate_epoch(
            model, val_loader, criterion, device, class_names,
            img_mean=img_mean, img_std=img_std
        )
        val_loss = val_metrics['loss']
        val_macro_f1 = val_metrics['macro_f1']
        
        if scheduler is not None:
            scheduler.step()
            
        epoch_time = time.time() - t0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_macro_f1'].append(train_metrics['macro_f1'])
        history['val_macro_f1'].append(val_macro_f1)
        history['epoch_times'].append(epoch_time)
        
        if verbose:
            print(f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_time:.1f}s] - "
                  f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['macro_f1']:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val F1: {val_macro_f1:.4f}")
                  
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_macro_f1,
                'metrics': {k: v for k, v in val_metrics.items() if k != 'probs'}
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch} with Val F1: {best_val_f1:.4f})")
                break
                
    total_time = time.time() - start_train_time
    history['total_time'] = total_time
    history['best_val_f1'] = best_val_f1
    history['best_epoch'] = best_epoch
    
    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

def run_sanity_check_overfit(model, dataloader, device, steps=80, save_path='artifacts/task1/fig_g1_sanity_overfit.png'):
    print("\n[Sanity Check] Running single-batch overfit test...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    batch = next(iter(dataloader))
    img_mean = dataloader.dataset.img_mean
    img_std = dataloader.dataset.img_std
    
    raw_images = batch['image'].to(device)
    images = (raw_images.float() / 255.0 - img_mean) / img_std
    sensors = batch['sensor'].to(device)
    masks = batch['sensor_mask'].to(device)
    targets = batch['label'].to(device)
    
    loss_history = []
    acc_history = []
    
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, sensors, masks)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == targets).float().mean().item() * 100
        loss_val = loss.item()
        loss_history.append(loss_val)
        acc_history.append(acc)
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1:02d}/{steps}: Loss = {loss_val:.5f}, Accuracy = {acc:.1f}%")
            
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(loss_history, color='#c0392b', lw=2, label='Cross-Entropy Loss')
    ax1.set_xlabel('Optimization Steps', fontsize=11)
    ax1.set_ylabel('Loss', color='#c0392b', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='#c0392b')
    ax1.set_title(f'Sanity Check: Overfitting Mini-batch (Final Loss = {loss_history[-1]:.4f}, Acc = {acc_history[-1]:.1f}%)',
                  fontsize=12, fontweight='bold', pad=10)
                  
    ax2 = ax1.twinx()
    ax2.plot(acc_history, color='#27ae60', lw=2, linestyle='--', label='Training Accuracy (%)')
    ax2.set_ylabel('Akurasi (%)', color='#27ae60', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax2.set_ylim(-5, 105)
    ax2.grid(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Sanity check complete! Plot saved to: {save_path}")
    return loss_history[-1]
