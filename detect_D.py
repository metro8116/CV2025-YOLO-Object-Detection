import json
import os
import cv2
from ultralytics import YOLO

# ------------------ 配置区 ------------------
STUDENT_NAME            = "刘崇轩"
SCORE_THRESH_MODEL1     = 0.9      # 模型1 置信度阈值
SCORE_THRESH_MODEL2     = 0.6      # 模型2 置信度阈值
YOLO_MODEL1             = "best1.pt"  # 模型1 权重文件
YOLO_MODEL2             = "best2.pt"  # 模型2 权重文件

# 模型1 只保留以下类别：
ALLOWED_CLASSES_MODEL1 = {
    0: "People",
    1: "Bike",
}
# 模型2 只保留以下类别：
ALLOWED_CLASSES_MODEL2 = {
    0: "Light",
    2: "Roadblock"
}

# 输出 JSON 时，指定新的 class_id
OUTPUT_ID_MAP = {
    "Light": 2,
    "Roadblock": 3
}

# 每个类别对应的可视化框颜色 (B, G, R)
BOX_COLOR_MAP = {
    "People":    (0, 255, 0),   # 绿色
    "Bike":      (255, 0, 0),   # 蓝色
    "Light":     (0, 0, 255),   # 红色
    "Roadblock": (0, 255, 255), # 黄色
}
# 每个类别对应的文字颜色 (B, G, R)
TEXT_COLOR_MAP = {
    "People":    (0, 0, 0),     # 黑色
    "Bike":      (255, 255, 255), # 白色
    "Light":     (255, 255, 255), # 白色
    "Roadblock": (0, 0, 0),     # 黑色
}

# 文本样式
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.6
THICKNESS       = 2

# 全局加载两个 YOLO 模型
model1 = YOLO(YOLO_MODEL1)
model2 = YOLO(YOLO_MODEL2)


def add_detection(results, label, output_id, xmin, ymin, xmax, ymax):
    """向 results 字典中添加一个检测框"""
    results["objects"].append({
        "label": label,
        "class_id": int(output_id),
        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
    })


def run_model(model, allowed_classes, thresh, image, vis, results):
    """
    使用指定模型检测并将符合阈值和类别限制的结果添加到 results 中，同时绘制到 vis 图像上。
    """
    preds = model(image)[0]
    boxes = preds.boxes.xyxy.cpu().numpy()
    confs = preds.boxes.conf.cpu().numpy()
    cls_ids = preds.boxes.cls.cpu().numpy().astype(int)

    for (xmin, ymin, xmax, ymax), conf, cls_id in zip(boxes, confs, cls_ids):
        if conf < thresh or cls_id not in allowed_classes:
            continue
        label = allowed_classes[cls_id]
        # 根据 label 映射新的 class_id
        output_id = OUTPUT_ID_MAP.get(label, cls_id)
        add_detection(results, label, output_id, xmin, ymin, xmax, ymax)

        # 框与文字颜色映射
        box_color = BOX_COLOR_MAP.get(label, (0, 255, 0))
        text_color = TEXT_COLOR_MAP.get(label, (0, 0, 0))

        # 画框
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        cv2.rectangle(vis, pt1, pt2, box_color, THICKNESS)

        # 文字背景
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
        cv2.rectangle(vis,
                      (pt1[0], pt1[1] - th - 8),
                      (pt1[0] + tw, pt1[1]),
                      box_color, -1)
        # 写文字
        cv2.putText(vis, text, (pt1[0], pt1[1] - 4),
                    FONT, FONT_SCALE, text_color, 1, cv2.LINE_AA)


def detector(image_path):
    """
    对单张图片使用两个模型分别检测指定类别，合并输出 JSON 和可视化图片。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] 无法读取图片: {image_path}")
        return

    image_name = os.path.basename(image_path)
    results = {"image_name": image_name, "objects": []}
    vis = image.copy()

    # 分别运行两个模型并合并结果
    run_model(model1, ALLOWED_CLASSES_MODEL1, SCORE_THRESH_MODEL1, image, vis, results)
    run_model(model2, ALLOWED_CLASSES_MODEL2, SCORE_THRESH_MODEL2, image, vis, results)

    # 保存 JSON 结果
    json_name = f"{STUDENT_NAME}_predicted_result.json"
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存合并检测结果: {json_name}")

    # 保存可视化图片
    vis_name = f"vis_{image_name}"
    cv2.imwrite(vis_name, vis)
    print(f"[INFO] 已保存可视化图片: {vis_name}")


if __name__ == "__main__":
    # 对当前文件夹下的 D_TOTAL.jpg 进行检测并合并输出
    detector("D_TOTAL.jpg")