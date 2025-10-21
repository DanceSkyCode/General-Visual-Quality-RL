import os
import numpy as np
from scipy.stats import spearmanr, pearsonr

gt_path = r""
pred_path = r""
gt_dict = {}
with open(gt_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        img_name = parts[1]
        mos = float(parts[2])
        gt_dict[img_name] = mos

print(f"✅ Ground Truth: {len(gt_dict)}")

pred_dict = {}
with open(pred_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name = parts[0]
        score = float(parts[1])
        pred_dict[img_name] = score

print(f"✅ Prediction: {len(pred_dict)}")

gt_scores = []
pred_scores = []
missing = []

for img_name, mos in gt_dict.items():
    if img_name in pred_dict:
        gt_scores.append(mos)
        pred_scores.append(pred_dict[img_name])
    else:
        missing.append(img_name)

print(f"⚙️ Success match: {len(gt_scores)} / {len(gt_dict)}")
if missing:
    print(f"⚠️ {len(missing)} find, miss: {missing[:5]}")
if len(gt_scores) > 1:
    srcc, _ = spearmanr(gt_scores, pred_scores)
    plcc, _ = pearsonr(gt_scores, pred_scores)

    print(f"📊 SRCC (Spearman Rank Correlation Coefficient): {srcc:.4f}")
    print(f"📈 PLCC (Pearson Linear Correlation Coefficient):  {plcc:.4f}")
else:
    print("❌")
