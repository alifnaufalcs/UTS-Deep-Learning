"""
Evaluation and Analysis Module for Task 1: Multimodal Defect Classification
High-performance vectorized evaluation with batch GPU normalization.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import calculate_metrics, plot_confusion_matrix

@torch.no_grad()
def evaluate_missing_sensor_degradation(model, val_loader, device, class_names,
                                        mask_rates=[0.0, 0.25, 0.50, 1.0],
                                        out_dir='artifacts/task1'):
    print("\n[Task f] Evaluating graceful degradation under artificial sensor masking...")
    model.eval()
    results = []
    use_cuda = device.type == 'cuda'
    img_mean = val_loader.dataset.img_mean
    img_std = val_loader.dataset.img_std
    
    for rate in mask_rates:
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            raw_images = batch['image'].to(device)
            sensors = batch['sensor'].clone().to(device)
            masks = batch['sensor_mask'].clone().to(device)
            targets = batch['label'].to(device)
            B = raw_images.size(0)
            
            images = (raw_images.float() / 255.0 - img_mean) / img_std
            
            if rate > 0.0:
                drop_mask = (torch.rand(B, 1, device=device) < rate).float()
                masks = masks * (1.0 - drop_mask)
                sensors = sensors * masks.unsqueeze(-1)
                
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits = model(images, sensors, masks)
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
        metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), class_names)
        results.append({
            'Persentase_Sensor_Dimasking': f"{int(rate * 100)}%",
            'Macro_F1': round(metrics['macro_f1'], 4),
            'Accuracy': round(metrics['accuracy'], 4),
            'Penurunan_F1_Relatif': round((results[0]['Macro_F1'] - metrics['macro_f1']) if len(results) > 0 else 0.0, 4)
        })
        print(f"  Masking {int(rate*100):3d}% -> Macro-F1 = {metrics['macro_f1']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
        
    df_results = pd.DataFrame(results)
    table_path = os.path.join(out_dir, 'table_missing_modality_eval.csv')
    df_results.to_csv(table_path, index=False)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates_num = [0, 25, 50, 100]
    f1_vals = [r['Macro_F1'] for r in results]
    ax.plot(rates_num, f1_vals, marker='o', lw=2.2, color='#2980b9', markersize=8)
    ax.set_title("Dampak Sensor Hilang terhadap Performa (Graceful Degradation)", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Persentase Sensor yang Hilang / Dimasking (%)", fontsize=11)
    ax.set_ylabel("Validation Macro-F1", fontsize=11)
    ax.set_ylim(min(f1_vals) - 0.05, max(f1_vals) + 0.05)
    for x_val, y_val in zip(rates_num, f1_vals):
        ax.annotate(f"{y_val:.4f}", (x_val, y_val), xytext=(0, 7), textcoords="offset points", ha='center', fontweight='semibold')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_f1_sensor_masking_drop.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Task (f) artifacts saved: {table_path}, {fig_path}")
    return df_results

@torch.no_grad()
def evaluate_per_line(model, val_loader, device, class_names, out_dir='artifacts/task1'):
    print("\n[Task e] Evaluating performance per production line (Lini A vs Lini B)...")
    model.eval()
    use_cuda = device.type == 'cuda'
    img_mean = val_loader.dataset.img_mean
    img_std = val_loader.dataset.img_std
    
    line_preds = {'A': [], 'B': []}
    line_targets = {'A': [], 'B': []}
    
    meta_df = val_loader.dataset.meta_df
    
    batch_start = 0
    for batch in val_loader:
        raw_images = batch['image'].to(device)
        sensors = batch['sensor'].to(device)
        masks = batch['sensor_mask'].to(device)
        targets = batch['label'].to(device)
        B = raw_images.size(0)
        
        images = (raw_images.float() / 255.0 - img_mean) / img_std
        
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(images, sensors, masks)
            
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        lines = meta_df.iloc[batch_start : batch_start + B]['lini'].values
        batch_start += B
        
        for p, t, l in zip(preds, targets_np, lines):
            line_preds[l].append(p)
            line_targets[l].append(t)
            
    rows = []
    for l in ['A', 'B']:
        m = calculate_metrics(np.array(line_targets[l]), np.array(line_preds[l]), class_names)
        rows.append({
            'Lini_Produksi': f"Lini {l}",
            'Jumlah_Sampel': len(line_targets[l]),
            'Macro_F1': round(m['macro_f1'], 4),
            'Accuracy': round(m['accuracy'], 4)
        })
        print(f"  Lini {l} (N={len(line_targets[l])}) -> Macro-F1 = {m['macro_f1']:.4f}, Accuracy = {m['accuracy']:.4f}")
        
    df_line = pd.DataFrame(rows)
    table_path = os.path.join(out_dir, 'table_domain_shift_line_eval.csv')
    df_line.to_csv(table_path, index=False)
    print(f"Task (e) artifact saved: {table_path}")
    return df_line

def detect_noisy_labels(model, dataloader, device, class_names, threshold_loss_percentile=95, out_dir='artifacts/task1'):
    print("\n[Task d] Detecting noisy/mislabeled training samples via loss audit...")
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    use_cuda = device.type == 'cuda'
    img_mean = dataloader.dataset.img_mean
    img_std = dataloader.dataset.img_std
    
    all_losses = []
    all_indices = []
    all_preds = []
    all_targets = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in dataloader:
            raw_images = batch['image'].to(device)
            sensors = batch['sensor'].to(device)
            masks = batch['sensor_mask'].to(device)
            targets = batch['label'].to(device)
            indices = batch['index']
            
            images = (raw_images.float() / 255.0 - img_mean) / img_std
            
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits = model(images, sensors, masks)
                loss = criterion(logits, targets)
                probs = torch.softmax(logits, dim=1)
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            confs = probs[torch.arange(len(targets)), targets].cpu().numpy()
            
            all_losses.extend(loss.cpu().numpy())
            all_indices.extend(indices.numpy() if isinstance(indices, torch.Tensor) else indices)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_confidences.extend(confs)
            
    losses = np.array(all_losses)
    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    confs_np = np.array(all_confidences)
    
    loss_threshold = np.percentile(losses, threshold_loss_percentile)
    noisy_mask = (losses >= loss_threshold) & (targets_np != preds_np)
    noisy_count = np.sum(noisy_mask)
    noisy_pct = (noisy_count / len(losses)) * 100
    
    print(f"  Total samples audited: {len(losses):,}")
    print(f"  Suspected mislabeled samples: {noisy_count:,} ({noisy_pct:.2f}%) at {threshold_loss_percentile}th percentile loss >= {loss_threshold:.4f}")
    
    suspected_indices = np.where(noisy_mask)[0]
    suspected_samples = pd.DataFrame({
        'sample_index': suspected_indices[:100],
        'given_label': [class_names[targets_np[i]] for i in suspected_indices[:100]],
        'model_predicted_label': [class_names[preds_np[i]] for i in suspected_indices[:100]],
        'given_label_confidence': confs_np[suspected_indices[:100]].round(4),
        'cross_entropy_loss': losses[suspected_indices[:100]].round(4)
    })
    
    table_path = os.path.join(out_dir, 'table_noisy_samples.csv')
    suspected_samples.to_csv(table_path, index=False)
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.histplot(losses, bins=50, kde=True, color='#2980b9', ax=ax)
    ax.axvline(loss_threshold, color='#e74c3c', linestyle='--', lw=2, label=f'Threshold Mislabel (Loss >= {loss_threshold:.2f})')
    ax.set_title(f"Distribusi Sample-wise Cross-Entropy Loss (Deteksi Noisy Labels ~{noisy_pct:.1f}%)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Cross-Entropy Loss", fontsize=10)
    ax.set_ylabel("Frekuensi", fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'fig_d1_loss_distribution_noise.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Task (d) artifacts saved: {table_path}, {fig_path}")
    return noisy_pct, suspected_samples

@torch.no_grad()
def generate_test_predictions(model, test_loader, test_meta, class_names,
                              out_csv='C:/DL_Classification/prediksi_klasifikasi.csv'):
    print(f"\n[Test Inference] Generating test predictions on {len(test_meta):,} units...")
    model.eval()
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    img_mean = test_loader.dataset.img_mean
    img_std = test_loader.dataset.img_std
    
    all_probs = []
    
    for batch in test_loader:
        raw_images = batch['image'].to(device)
        sensors = batch['sensor'].to(device)
        masks = batch['sensor_mask'].to(device)
        
        images = (raw_images.float() / 255.0 - img_mean) / img_std
        
        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(images, sensors, masks)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        all_probs.append(probs)
        
    all_probs = np.vstack(all_probs)
    predicted_indices = np.argmax(all_probs, axis=1)
    predicted_classes = [class_names[idx] for idx in predicted_indices]
    
    df_pred = pd.DataFrame({
        'id_unit': test_meta['id_unit'],
        'kelas': predicted_classes
    })
    
    df_pred.to_csv(out_csv, index=False)
    print(f"Test predictions successfully written to: {out_csv}")
    
    validate_submission_file(out_csv, test_meta, class_names)
    return df_pred, all_probs

def validate_submission_file(csv_path, test_meta, class_names):
    print(f"\n[Submission Validator] Running strict verification on {csv_path}...")
    assert os.path.exists(csv_path), f"File not found: {csv_path}"
    
    df = pd.read_csv(csv_path)
    print(f"  Rows read: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")
    
    assert list(df.columns) == ['id_unit', 'kelas'], f"Header mismatch! Expected ['id_unit', 'kelas'], got {list(df.columns)}"
    assert len(df) == len(test_meta), f"Row count mismatch! Expected {len(test_meta)}, got {len(df)}"
    assert df['id_unit'].nunique() == len(test_meta), "Duplicate id_unit found!"
    assert (df['id_unit'].values == test_meta['id_unit'].values).all(), "id_unit order or values do not match test_meta!"
    
    valid_classes = set(class_names)
    actual_classes = set(df['kelas'].unique())
    invalid = actual_classes - valid_classes
    assert len(invalid) == 0, f"Found invalid class names: {invalid}"
    
    print("  Class distribution in test predictions:")
    for k, v in df['kelas'].value_counts().items():
        print(f"    {k:<12}: {v:,} ({v/len(df)*100:.1f}%)")
        
    print("[Submission Validator] PASS! All checks passed successfully.")
