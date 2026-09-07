import os
import argparse
import math
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLDS = [0.5]  
LINE_WIDTH = 30
IMG_SHAPE = (720, 1366)

def remove_consecutive_duplicates(points):
    if len(points) < 2:
        return points
    cleaned = [points[0]]
    for p in points[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    return cleaned

def parse_lines_txt(txt_path):
    # 遇到任何格式错误直接抛出异常
    lanes = []
    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
        return lanes
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            coords = line.split()
            if len(coords) % 2 != 0:
                raise ValueError(f"文件 {txt_path} 第 {line_idx+1} 行数值数量为奇数")
            if len(coords) < 4:
                raise ValueError(f"文件 {txt_path} 第 {line_idx+1} 行少于4个数值")
            
            pts = []
            for i in range(0, len(coords), 2):
                x, y = float(coords[i]), float(coords[i+1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"文件 {txt_path} 第 {line_idx+1} 行包含 NaN 或 Inf")
                pts.append((x, y))
            
            cleaned_pts = remove_consecutive_duplicates(pts)
            if len(cleaned_pts) < 2:
                raise ValueError(f"文件 {txt_path} 第 {line_idx+1} 行去重后不足2个有效点")
                
            lanes.append(cleaned_pts)
    return lanes

def interp_lane(points, n=5):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    k = min(3, len(points) - 1)
    tck, u = splprep([x, y], s=0, t=n, k=k)
    u_new = np.linspace(0., 1., num=(len(u) - 1) * n + 1)
    return np.array(splev(u_new, tck)).T

def draw_lane_mask(lane, width):
    img = np.zeros(IMG_SHAPE, dtype=np.uint8)
    lane = np.asarray(lane, dtype=np.float64)
    lane[:, 0] = np.clip(lane[:, 0], 0, IMG_SHAPE[1] - 1) 
    lane[:, 1] = np.clip(lane[:, 1], 0, IMG_SHAPE[0] - 1) 
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(1,), thickness=width)
    return img > 0

def culane_metric_single(pred, anno, width=LINE_WIDTH, iou_thresholds=IOU_THRESHOLDS):
    metric_res = {}
    if len(pred) == 0 or len(anno) == 0:
        for thr in iou_thresholds:
            metric_res[thr] = [0, len(pred), len(anno)] # [tp, fp, fn]
        return metric_res

    interp_pred = [interp_lane(lane) for lane in pred]
    interp_anno = [interp_lane(lane) for lane in anno]

    ious = np.zeros((len(pred), len(anno)))
    pred_masks = [draw_lane_mask(p, width) for p in interp_pred]
    anno_masks = [draw_lane_mask(a, width) for a in interp_anno]
    
    for i, p_mask in enumerate(pred_masks):
        for j, a_mask in enumerate(anno_masks):
            union = (p_mask | a_mask).sum()
            if union > 0:
                ious[i, j] = (p_mask & a_mask).sum() / union

    row_ind, col_ind = linear_sum_assignment(1 - ious)

    for thr in iou_thresholds:
        tp = int((ious[row_ind, col_ind] > thr).sum())
        fp = len(pred) - tp
        fn = len(anno) - tp
        metric_res[thr] = [tp, fp, fn]
        
    return metric_res

def eval_predictions(pred_dir, gt_dir, list_path, width=LINE_WIDTH, iou_thresholds=IOU_THRESHOLDS):
    tasks = []
    with open(list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('/')
            folder_name = parts[-2]
            file_name = parts[-1].replace('.jpg', '.lines.txt')
            
            rel_path = os.path.join(folder_name, file_name)
            tasks.append((os.path.join(pred_dir, rel_path), os.path.join(gt_dir, rel_path)))
            
    results = []
    for pred_f, gt_f in tasks:
        # 解析失败直接抛出异常中断评分
        pred_data = parse_lines_txt(pred_f)
        anno_data = parse_lines_txt(gt_f)
        results.append(culane_metric_single(pred_data, anno_data, width, iou_thresholds))

    ret = {}
    mean_f1 = 0
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        
        precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        mean_f1 += f1 / len(iou_thresholds)
        ret[thr] = {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}
        print(f"Thr: {thr} | F1: {f1:.4f} | P: {precision:.4f} | R: {recall:.4f}")
        
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--list_path', type=str, required=True)
    args = parser.parse_args()
    
    eval_predictions(args.pred_dir, args.gt_dir, args.list_path)