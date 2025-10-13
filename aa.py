import os
import shutil
import numpy as np
import cv2
import random
from ultralytics import YOLO  # YOLOv11官方库

# ==============================================================================
# 配置参数（需手动确认！确保路径正确）
# ==============================================================================
# 1. 数据路径
ORIGINAL_ROOT_DIR = "未标注的数据"  # 原始数据根目录（含a_01等子文件夹）
SAVE_RESULT_DIR = "yolov11_data_results"  # 处理后的数据保存目录（自动创建）
# 2. 本地YOLO模型路径（确保yolov11n.pt在该路径下！）
LOCAL_YOLO_MODEL_PATH = "yolo11n.pt"  # 若模型不在当前目录，填绝对路径（如"C:/xxx/yolo11n.pt"）
# 3. 训练参数（CPU训练建议batch=2，GPU可设4-8）
TRAIN_EPOCHS = 30        # 训练轮次（数据少设20-30，多设50-100）
TRAIN_BATCH_SIZE = 2     # 批次大小（CPU必改2！8GB GPU→4，16GB GPU→8）
TRAIN_IMG_SIZE = 640     # 图片尺寸（YOLOv11默认640，无需修改）
TRAIN_DEVICE = "cpu"     # 0=第1块GPU，"cpu"=用CPU训练（无GPU时保持）


# ==============================================================================
# 前3步：数据清洗、去重与RGB图合成（无网络依赖）
# ==============================================================================
def step1_to_step3_clean_data():
    """仅处理本地数据，无任何网络请求"""
    # 创建结果目录（不存在则创建）
    os.makedirs(SAVE_RESULT_DIR, exist_ok=True)
    total_label_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    valid_full_names = []

    # 遍历原始数据的子文件夹（如a_01、a_02）
    for subdir_name in os.listdir(ORIGINAL_ROOT_DIR):
        subdir_path = os.path.join(ORIGINAL_ROOT_DIR, subdir_name)
        if not os.path.isdir(subdir_path):
            print(f"⚠️  {subdir_path}不是文件夹，跳过")
            continue
        print(f"✅ 正在处理子文件夹：{subdir_name}")

        # 定位核心数据目录（必须包含label、pl_image、surface_image）
        train_data_path = os.path.join(subdir_path, "train_data")
        label_dir = os.path.join(train_data_path, "label")
        pl_img_dir = os.path.join(train_data_path, "pl_image")
        surface_img_dir = os.path.join(train_data_path, "surface_image")

        # 检查核心目录是否存在，缺少则跳过
        if not all(os.path.exists(x) for x in [label_dir, pl_img_dir, surface_img_dir]):
            print(f"⚠️  子文件夹{subdir_name}缺少核心目录（label/pl_image/surface_image），跳过")
            continue

        # 处理每个标签文件（仅保留txt格式）
        for txt_filename in os.listdir(label_dir):
            if not txt_filename.lower().endswith(".txt"):
                print(f"⚠️  {txt_filename}不是txt文件，跳过")
                continue

            # 修复双重后缀问题（如"000_1_2.txt.txt" → "000_1_2"）
            original_prefix = os.path.splitext(txt_filename)[0]
            if original_prefix.lower().endswith(".txt"):
                original_prefix = os.path.splitext(original_prefix)[0]
                print(f"ℹ️  修复双重后缀：{txt_filename} → {original_prefix}.txt")

            # 生成唯一文件名（子文件夹名_原前缀，避免不同子文件夹重名）
            full_prefix = f"{subdir_name}_{original_prefix}"
            original_txt_path = os.path.join(label_dir, txt_filename)

            # 清洗标签：只保留1-6类的有效标注（过滤无效类别）
            cleaned_lines = []
            try:
                with open(original_txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped_line = line.strip()
                        if not stripped_line:
                            continue  # 跳过空行
                        # 检查是否以1-6开头（有效标签格式：类别 x y w h）
                        if stripped_line[0].isdigit() and 1 <= int(stripped_line[0]) <= 6:
                            cleaned_lines.append(stripped_line)
                            total_label_count[int(stripped_line[0])] += 1
            except Exception as e:
                print(f"⚠️  读取{txt_filename}失败：{str(e)}，跳过")
                continue

            # 无有效标签的文件不保存，直接跳过
            if not cleaned_lines:
                print(f"⚠️  {txt_filename}无有效标签（仅保留1-6类），跳过")
                continue

            # 保存清洗后的标签到结果目录
            save_txt_path = os.path.join(SAVE_RESULT_DIR, f"{full_prefix}.txt")
            try:
                with open(save_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(cleaned_lines))
            except Exception as e:
                print(f"⚠️  保存{full_prefix}.txt失败：{str(e)}，跳过")
                continue

            # 匹配对应图片（PL图和Surface图，支持png/jpg/jpeg/bmp格式）
            image_exts = [".png", ".jpg", ".jpeg", ".bmp"]
            pl_img_path = None
            surface_img_path = None

            # 自动匹配图片格式（找到任一格式即可）
            for ext in image_exts:
                if not pl_img_path:
                    test_pl = os.path.join(pl_img_dir, f"{original_prefix}{ext}")
                    if os.path.exists(test_pl):
                        pl_img_path = test_pl
                if not surface_img_path:
                    test_surface = os.path.join(surface_img_dir, f"{original_prefix}{ext}")
                    if os.path.exists(test_surface):
                        surface_img_path = test_surface
                if pl_img_path and surface_img_path:
                    break  # 两者都找到则退出循环

            # 缺少任一图片则清理已保存的标签，跳过该组数据
            if not pl_img_path or not surface_img_path:
                print(f"⚠️  {full_prefix}缺少图片（PL图：{bool(pl_img_path)}，Surface图：{bool(surface_img_path)}），跳过")
                os.remove(save_txt_path)
                continue

            # 读取图片并强制转为2维灰度图（解决维度不匹配问题）
            def read_gray_img(img_path):
                """读取图片并确保返回2维灰度图（避免cv2.merge报错）"""
                img = cv2.imread(img_path)  # 先按彩色图读取（兼容更多格式）
                if img is None:
                    return None
                # 3维（RGB/BGR）→ 转灰度
                if len(img.shape) == 3:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 4维（带Alpha通道）→ 转灰度
                elif len(img.shape) == 4:
                    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                # 已为2维（纯灰度图）→ 直接返回
                else:
                    return img

            # 读取并校验图片有效性
            pl_img = read_gray_img(pl_img_path)
            surface_img = read_gray_img(surface_img_path)
            if pl_img is None or surface_img is None:
                print(f"⚠️  {full_prefix}图片读取失败（可能损坏），跳过")
                os.remove(save_txt_path)
                continue
            if pl_img.size == 0 or surface_img.size == 0:
                print(f"⚠️  {full_prefix}图片尺寸为空，跳过")
                os.remove(save_txt_path)
                continue

            # 统一图片尺寸（以PL图为标准，避免标注坐标偏移）
            target_height, target_width = pl_img.shape  # PL图尺寸（高，宽）
            if surface_img.shape != (target_height, target_width):
                # 缩放Surface图到PL图尺寸（线性插值，兼顾清晰度）
                surface_img = cv2.resize(
                    surface_img,
                    (target_width, target_height),  # cv2.resize参数：(宽，高)
                    interpolation=cv2.INTER_LINEAR
                )
                print(f"ℹ️  {full_prefix}：Surface图尺寸统一为{target_width}×{target_height}（与PL图一致）")

            # 合成RGB图（R=PL，G=PL，B=Surface）并保存
            try:
                rgb_image = cv2.merge([pl_img, pl_img, surface_img])
                save_rgb_path = os.path.join(SAVE_RESULT_DIR, f"{full_prefix}.png")
                cv2.imwrite(save_rgb_path, rgb_image)

                # 验证图片保存结果（避免空文件或损坏）
                if not os.path.exists(save_rgb_path) or os.path.getsize(save_rgb_path) < 100:
                    raise Exception("图片保存后为空或过小（<100字节）")

                valid_full_names.append(full_prefix)
                print(f"✅ 成功处理：{full_prefix}（标签+图片）")
            except Exception as e:
                print(f"⚠️  保存{full_prefix}.png失败：{str(e)}，跳过")
                # 清理已保存的标签和异常图片
                os.remove(save_txt_path)
                if os.path.exists(save_rgb_path):
                    os.remove(save_rgb_path)
                continue

    # 输出前3步处理汇总报告
    print("\n" + "="*60)
    print("📊 前3步数据处理完成！")
    print(f"1-6类标签总数：{total_label_count}")
    print(f"有效文件总数（图+标匹配）：{len(valid_full_names)} 个")
    print(f"处理后数据保存路径：{os.path.abspath(SAVE_RESULT_DIR)}")
    print("="*60 + "\n")

    return total_label_count, valid_full_names


# ==============================================================================
# 第4步：YOLOv11训练（仅用本地模型，无网络请求）
# ==============================================================================
def step4_train_yolov11():
    """仅加载本地YOLO模型，不发起任何网络请求"""
    # 1. 检查本地模型文件是否存在
    if not os.path.exists(LOCAL_YOLO_MODEL_PATH):
        print(f"❌ 本地模型文件不存在！请将yolov11n.pt放到以下路径：")
        print(f"   {os.path.abspath(LOCAL_YOLO_MODEL_PATH)}")
        print("   模型下载地址：https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n.pt")
        return
    print(f"✅ 本地模型文件已找到：{os.path.abspath(LOCAL_YOLO_MODEL_PATH)}")

    # 2. 检查处理后的数据是否有效（必须有图片和标签）
    all_imgs = [f for f in os.listdir(SAVE_RESULT_DIR) if f.lower().endswith(".png")]
    all_labels = [f for f in os.listdir(SAVE_RESULT_DIR) if f.lower().endswith(".txt")]
    if not all_imgs:
        print(f"❌ 处理后的数据目录中无PNG图片，请先确保step1_to_step3_clean_data()正常执行！")
        return
    if not all_labels:
        print(f"❌ 处理后的数据目录中无TXT标签，请先确保step1_to_step3_clean_data()正常执行！")
        return

    # 3. 匹配图片-标签对（确保文件名一一对应）
    img_prefixes = [os.path.splitext(f)[0] for f in all_imgs]
    label_prefixes = [os.path.splitext(f)[0] for f in all_labels]
    valid_prefixes = list(set(img_prefixes) & set(label_prefixes))  # 取交集（图+标都存在）

    if not valid_prefixes:
        print(f"❌ 未找到匹配的图片-标签对（文件名不统一）！")
        print(f"   正确格式示例：图片名=a_01_000_1_2.png，标签名=a_01_000_1_2.txt")
        return
    print(f"✅ 找到 {len(valid_prefixes)} 对有效图片-标签数据")

    # 4. 划分训练集/验证集（8:2比例，随机打乱避免类别集中）
    random.shuffle(valid_prefixes)
    split_idx = int(len(valid_prefixes) * 0.8)
    train_prefixes = valid_prefixes[:split_idx]  # 80%训练集
    val_prefixes = valid_prefixes[split_idx:]    # 20%验证集
    if len(val_prefixes) == 0:  # 若数据过少（<5个），强制留1个验证集
        val_prefixes = [train_prefixes.pop()]

    # 5. 生成YOLO所需的训练/验证文件列表（关键：写入图片绝对路径）
    # 训练集列表（每行是图片的完整绝对路径）
    train_list_path = os.path.join(SAVE_RESULT_DIR, "train.txt")
    with open(train_list_path, "w", encoding="utf-8") as f:
        for prefix in train_prefixes:
            img_abs_path = os.path.abspath(os.path.join(SAVE_RESULT_DIR, f"{prefix}.png"))
            f.write(f"{img_abs_path}\n")
    # 验证集列表（同样写入绝对路径）
    val_list_path = os.path.join(SAVE_RESULT_DIR, "val.txt")
    with open(val_list_path, "w", encoding="utf-8") as f:
        for prefix in val_prefixes:
            img_abs_path = os.path.abspath(os.path.join(SAVE_RESULT_DIR, f"{prefix}.png"))
            f.write(f"{img_abs_path}\n")
    print(f"✅ 生成训练集列表：{os.path.abspath(train_list_path)}（{len(train_prefixes)}张图）")
    print(f"✅ 生成验证集列表：{os.path.abspath(val_list_path)}（{len(val_prefixes)}张图）")

    # 6. 生成YOLO训练配置文件（custom.yaml，修正类别数为6）
    custom_yaml_path = os.path.join(SAVE_RESULT_DIR, "custom.yaml")
    abs_save_dir = os.path.abspath(SAVE_RESULT_DIR)
    yaml_content = f"""
path: {abs_save_dir}  # 数据根目录（绝对路径，格式兼容）
train: train.txt      # 训练集列表（已包含绝对路径，无需额外拼接）
val: val.txt          # 验证集列表（已包含绝对路径，无需额外拼接）

nc: 6                 # 类别数量（1-6共6类，与标签严格对应）
names: ['1', '2', '3', '4', '5', '6']  # 类别名称（顺序需与标签数字一致）
    """.strip()

    # 写入配置文件
    with open(custom_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"✅ 生成YOLO训练配置文件：{os.path.abspath(custom_yaml_path)}")

    # 7. 加载本地模型并启动训练（核心步骤）
    print("\n" + "="*60)
    print("🚀 启动YOLOv11本地训练！")
    print(f"训练参数：轮次={TRAIN_EPOCHS} | 批次={TRAIN_BATCH_SIZE} | 图片尺寸={TRAIN_IMG_SIZE} | 设备={TRAIN_DEVICE}")
    print(f"训练集：{len(train_prefixes)} 张 | 验证集：{len(val_prefixes)} 张")
    print("="*60)

    try:
        # 加载本地YOLO模型（无网络请求，仅读取本地文件）
        model = YOLO(LOCAL_YOLO_MODEL_PATH)
        # 启动训练（结果保存在本地 runs/train/custom_yolov11 目录）
        model.train(
            data=custom_yaml_path,    # 配置文件绝对路径
            epochs=TRAIN_EPOCHS,      # 训练轮次
            batch=TRAIN_BATCH_SIZE,   # 批次大小（CPU必为2，避免内存溢出）
            imgsz=TRAIN_IMG_SIZE,     # 输入图片尺寸
            name="custom_yolov11",    # 训练结果子目录名
            device=TRAIN_DEVICE,      # 训练设备（cpu/GPU）
            verbose=True,             # 显示详细训练日志（便于排查问题）
            pretrained=True,          # 使用预训练权重（本地模型已包含，无需下载）
            weight_decay=0.0005,      # 权重衰减（防止过拟合）
            patience=10               # 早停机制（10轮无提升则停止，节省时间）
        )

        # 训练完成提示（结果路径）
        result_path = os.path.join(os.getcwd(), "runs", "train", "custom_yolov11")
        best_model_path = os.path.join(result_path, "weights", "best.pt")
        print(f"\n🎉 训练完成！所有结果保存在本地：")
        print(f"   训练日志路径：{os.path.abspath(result_path)}")
        print(f"   最佳模型文件：{os.path.abspath(best_model_path)}")
        print(f"   提示：best.pt可直接用于后续推理预测，无需重新训练！")
    except Exception as e:
        print(f"\n❌ 训练过程出错：{str(e)}")
        # 常见错误解决方案提示（针对性指导）
        if "CUDA out of memory" in str(e):
            print("💡 解决方案：当前用GPU训练但显存不足，可改为CPU训练（TRAIN_DEVICE='cpu'），或减小批次（TRAIN_BATCH_SIZE=2）")
        elif "invalid device ordinal" in str(e):
            print("💡 解决方案：无可用GPU设备，将TRAIN_DEVICE改为'train_device=\"cpu\"'")
        elif "No labels found" in str(e):
            print("💡 解决方案：检查标签文件内容，确保每行格式为「类别 x y w h」（类别1-6，坐标0-1之间）")
        elif "could not find image" in str(e):
            print("💡 解决方案：检查train.txt/val.txt中的路径是否正确，确保图片文件存在且路径无中文/空格")


# ==============================================================================
# 执行入口（按顺序运行：先处理数据 → 再训练模型）
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("📋 开始执行本地数据处理与YOLOv11训练（全程无网络）")
    print("="*60)


    step1_to_step3_clean_data()


    step4_train_yolov11()