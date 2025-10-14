import os
import csv
import numpy as np
from ultralytics import YOLO

def final_align_confusion(
    val_data_dir,  # YAML里val字段的路径（混淆矩阵统计的验证集）
    weight_path,   # 训练输出的best.pt路径（如runs/train/custom_yolov11/weights/best.pt）
    csv_out,
    nc         # 你的类别数（和YAML里nc一致）
):
    # 1. 加载模型，强制对齐混淆矩阵默认参数（和model.train()无关）
    model = YOLO(weight_path)
    model.conf = 0.001  # 固定对齐混淆矩阵阈值
    model.iou = 0.65    # 固定对齐混淆矩阵IOU
    model.max_det = 300 # 足够大，避免漏预测

    # 2. 获取YAML验证集里的所有图片（只统计这个路径下的图，和训练一致）
    val_imgs = []
    for img in os.listdir(val_data_dir):
        if img.endswith(".png"):  # 按你的图片格式改（如jpg）
            val_imgs.append(os.path.join(val_data_dir, img))
    if not val_imgs:
        raise FileNotFoundError(f"YAML验证集路径{val_data_dir}中无图片")

    # 3. 计算混淆矩阵+记录错误
    conf_matrix = np.zeros((nc, nc), dtype=int)
    error_records = []
    for img_path in val_imgs:
        img_name = os.path.basename(img_path)
        txt_path = os.path.join(val_data_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(txt_path):
            continue

        # 读真实目标（转0-5，对应1-6）
        true_cls = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and line[0].isdigit():
                    cls = int(line.split()[0]) - 1
                    if 0 <= cls < nc:
                        true_cls.append(cls)
        if not true_cls:
            continue

        # 模型推理（和混淆矩阵计算逻辑一致）
        results = model(img_path, verbose=False)[0]
        pred_cls = results.boxes.cls.numpy().astype(int) if len(results.boxes) > 0 else np.array([])
        pred_conf = results.boxes.conf.numpy() if len(results.boxes) > 0 else np.array([])

        # 匹配真实与预测（复刻YOLO逻辑）
        if len(pred_cls) > 0:
            sort_idx = np.argsort(pred_conf)[::-1]
            pred_cls = pred_cls[sort_idx]
        used_pred = [False] * len(pred_cls)

        # 统计错误（分错+漏检）
        for idx, t_cls in enumerate(true_cls, 1):
            matched = False
            for p_idx, p_cls in enumerate(pred_cls):
                if not used_pred[p_idx] and p_cls == t_cls:
                    conf_matrix[t_cls, p_cls] += 1
                    used_pred[p_idx] = True
                    matched = True
                    break
            if not matched:
                conf_matrix[t_cls, nc-1] += 1
                error_records.append([
                    os.path.abspath(img_path), img_name, idx,
                    str(t_cls+1), "无预测（漏检）", "漏检"
                ])
            else:
                # 检查是否分错（上面匹配到的是正确，这里找其他预测）
                for p_idx, p_cls in enumerate(pred_cls):
                    if not used_pred[p_idx] and p_cls != t_cls:
                        conf_matrix[t_cls, p_cls] += 1
                        error_records.append([
                            os.path.abspath(img_path), img_name, idx,
                            str(t_cls+1), str(p_cls+1), "标签分错"
                        ])
                        used_pred[p_idx] = True
                        break

    # 写入CSV+统计
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["图片路径", "图片名", "目标序号", "真实类别", "预测结果", "错误类型"])
        writer.writerows(error_records)

    # 混淆矩阵统计（和训练时输出的一致）
    total_true = conf_matrix.sum(axis=1)[:-1].sum()
    total_correct = np.trace(conf_matrix)
    total_error = total_true - total_correct

    print("="*80)
    print(f"✅ 对齐完成！基于你训练时YAML的val路径统计")
    print(f"📊 混淆矩阵：总真实目标{total_true} | 正确{total_correct} | 错误{total_error}")
    print(f"📁 CSV路径：{os.path.abspath(csv_out)} | 错误记录数{len(error_records)}")
    print(f"💡 若数量一致，说明完全对齐；若有差异，仅为同图多错误重复记录")
    print("="*80)

    # 1. YAML里val字段的路径（从你的custom_yaml_path文件里复制！）
    VAL_DATA_DIR = "D:/data/val"  # 比如你YAML里val

    # 2. 训练输出的best.pt路径（你的训练结果存在runs/train/custom_yolov11里）
    WEIGHT_PATH = "best.pt"

    # 3. 想保存的CSV文件名
    CSV_OUT = "yolov10_aligned_errors.csv"

    # 调用函数
    final_align_confusion(VAL_DATA_DIR, WEIGHT_PATH, CSV_OUT,7)