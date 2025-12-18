import json
import os
import cv2
from ultralytics import YOLO

# ------------------ 配置区 ------------------
STUDENT_NAME      = "刘崇轩"
SCORE_THRESH      = 0.125       # 置信度阈值
YOLO_MODEL        = "yolo11x.pt" # 模型权重文件
# 只保留以下 COCO 类别：
#   0: person -> "People"
ALLOWED_CLASS_IDS = {
    0: "People",
}
# 可视化时框和文字的样式
BOX_COLOR   = (0, 255, 0)       # 绿色框 (B,G,R)
TEXT_COLOR  = (0, 0, 0)         # 黑色文字
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.6
THICKNESS   = 2
# --------------------------------------------

# 全局加载 YOLO 模型
model = YOLO(YOLO_MODEL)

def add_detection(results, label, xmin, ymin, xmax, ymax):
    """向 results 字典中添加一个检测框"""
    results["objects"].append({
        "label": label,
        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
    })

def detector(image_path):
    """
    对单张图片使用 YOLO11x 检测 People 类，
    保存 JSON 并可视化输出带框的图片。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] 无法读取图片: {image_path}")
        return

    h, w = image.shape[:2]
    image_name = os.path.basename(image_path)
    results = {
        "image_name": image_name,
        "objects": []
    }

    # 推理
    preds = model(image, conf=SCORE_THRESH)[0]
    boxes_xyxy  = preds.boxes.xyxy.cpu().numpy()
    confidences = preds.boxes.conf.cpu().numpy()
    class_ids   = preds.boxes.cls.cpu().numpy().astype(int)

    # 可视化用的画布
    vis = image.copy()

    # 遍历所有检测框
    for (xmin, ymin, xmax, ymax), conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
        conf = float(conf)
        cls_id = int(cls_id)
        if conf < SCORE_THRESH or cls_id not in ALLOWED_CLASS_IDS:
            continue

        label = ALLOWED_CLASS_IDS[cls_id]
        # 添加到 JSON 结果
        add_detection(results, label, xmin, ymin, xmax, ymax)

        # 在 vis 图上画框
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        cv2.rectangle(vis, pt1, pt2, BOX_COLOR, THICKNESS)

        # 在框上方画文字背景
        text = f"{label} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
        cv2.rectangle(vis,
                      (pt1[0], pt1[1] - text_h - 8),
                      (pt1[0] + text_w, pt1[1]),
                      BOX_COLOR, -1)
        # 写文字
        cv2.putText(vis, text, (pt1[0], pt1[1] - 4),
                    FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)

    # 保存 JSON
    json_name = f"{STUDENT_NAME}_predicted_result.json"
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已保存检测结果: {json_name}")

    # 保存可视化图片
    vis_name = f"vis_{image_name}"
    cv2.imwrite(vis_name, vis)
    print(f"[INFO] 已保存可视化图片: {vis_name}")

if __name__ == "__main__":
    # 对当前文件夹下的 S_TOTAL.jpg 进行检测并可视化
    detector("S_TOTAL.jpg")
