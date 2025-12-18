# YOLOv11 Object Detection | 目标检测与模型融合

## 项目简介
本项目是南开大学 2025 年智能科学与技术课程设计（Part 2）。
项目使用 **Python** 和 **Ultralytics YOLO** 框架，对比了单模型检测与多模型融合检测在不同场景下的表现，并实现了自定义的 JSON 结果输出与可视化绘制。

## 实验内容

### 1. 单模型检测 (S_TOTAL)
- **场景**：体育馆内行人检测。
- **策略**：使用原始 `yolo11x.pt` 模型。
- **参数**：设置较低的置信度阈值 (`conf=0.125`) 以提高召回率 (Recall)。
- **输出**：检测所有 People 类别。

### 2. 多模型融合检测 (D_TOTAL)
- **场景**：复杂的校园道路场景。
- **策略**：**模型融合 (Ensemble)**。
    - 模型 A (`best1.pt`): 专注于检测 **People** 和 **Bike** (阈值 0.9)。
    - 模型 B (`best2.pt`): 专注于检测 **Light** 和 **Roadblock** (阈值 0.6)。
- **优势**：结合了不同微调模型的优势，同时兼顾了高精度与多类别覆盖。

## 文件结构

```text
├── detect_S.py             # 场景1：单模型检测脚本
├── detect_D.py             # 场景2：多模型融合检测脚本
├── yolo11x.pt              # 预训练权重
├── best1.pt                # 自定义权重 1
├── best2.pt                # 自定义权重 2
├── *_predicted_result.json # (生成的) 检测结果数据
├── *.jpg                   # 原图
├── vis_*.jpg               # (生成的) 可视化结果图
└── README.md
```

## 快速开始

### 1. 安装依赖
```bash
pip install ultralytics opencv-python
```

### 2. 运行单模型检测 (S_TOTAL)
```bash
python detect_S.py
# 输出：生成 vis_S_TOTAL.jpg 和对应的 JSON 文件
```

### 3. 运行多模型融合 (D_TOTAL)
```bash
python detect_D.py
# 输出：生成 vis_D_TOTAL.jpg 和对应的 JSON 文件
```

