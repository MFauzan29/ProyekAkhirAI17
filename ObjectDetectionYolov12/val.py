from ultralytics import YOLO
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# --- CONFIG ---
MODEL_PATH = "yolov12s_trained.pt"  # Path to your trained model
VAL_IMAGES = "dataset/coco128/images/train2017/*.jpg"  # Path to validation images
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Only these classes will be shown in plots
SELECTED_CLASSES = ['horse', 'car', 'truck', 'airplane']
SELECTED_IDS = [COCO_CLASSES.index(cls) for cls in SELECTED_CLASSES]

def main():
    model = YOLO(MODEL_PATH)
    image_paths = glob.glob(VAL_IMAGES)
    confidences = []
    classes = []
    inference_times = []
    metrics = model.val(data="coco128.yaml")

    print(f"Running inference on {len(image_paths)} images...")
    for img_path in image_paths:
        start = time.time()
        results = model(img_path)
        elapsed = (time.time() - start) * 1000  # ms
        inference_times.append(elapsed)
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                # Filter only selected classes
                for c, cf in zip(cls, conf):
                    if int(c) in SELECTED_IDS:
                        confidences.append(cf)
                        classes.append(int(c))

    confidences = np.array(confidences)
    classes = np.array(classes)
    inference_times = np.array(inference_times)

    # Print mAP@0.5 (map50), mAP@0.5:0.95 (map), Precision, Recall, dan "Accuracy"
    print(f"mAP@0.5:     {metrics.box.map50:.3f}")
    print(f"mAP@0.5:0.95:{metrics.box.map:.3f}")
    print(f"Precision:   {metrics.box.mp:.3f}")
    print(f"Recall:      {metrics.box.mr:.3f}")
    print(f"Accuracy:    {metrics.box.mp:.3f}")  # Accuracy diambil dari mean precision
    print(f"Total Inference Time: {np.sum(inference_times):.2f} ms")

    # --- Plot Model Performance Metrics ---
    plt.figure(figsize=(12, 4))
    
    # Performance metrics
    metrics_names = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'Accuracy']
    metrics_values = [metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr, metrics.box.mp]
    colors = ['skyblue', 'lightgreen', 'coral', 'gold', 'plum']
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.3f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Total inference time comparison
    plt.subplot(1, 2, 2)
    time_data = ['Total Time', 'Avg per Image']
    time_values = [np.sum(inference_times), np.mean(inference_times)]
    colors_time = ['red', 'orange']
    
    bars = plt.bar(time_data, time_values, color=colors_time, alpha=0.7, edgecolor='black')
    plt.title("Inference Time Summary")
    plt.ylabel("Time (ms)")
    
    # Add value labels on bars
    for bar, val in zip(bars, time_values):
        plt.text(bar.get_x() + bar.get_width()/2, val + max(time_values)*0.01, f"{val:.1f}", 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("model_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- Plot 1: Overall Confidence Distribution ---
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(confidences, bins=20, alpha=0.5, color='g', edgecolor='black')
    plt.title("Overall Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")

    # --- Plot 2: Average Confidence by Class ---
    class_conf = defaultdict(list)
    for c, conf in zip(classes, confidences):
        class_conf[c].append(conf)
    avg_conf = {c: np.mean(class_conf[c]) for c in class_conf}
    class_ids = [cid for cid in SELECTED_IDS if cid in class_conf]
    avg_vals = [avg_conf[c] for c in class_ids]
    class_names = [COCO_CLASSES[c] for c in class_ids]

    plt.subplot(1, 3, 2)
    bars = plt.bar(class_names, avg_vals, color='coral', alpha=0.7)
    plt.title("Average Confidence by Class")
    plt.ylabel("Average Confidence")
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, avg_vals):
        plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2f}", ha='center', va='bottom', fontweight='bold')

    # --- Plot 3: Confidence Distribution by Class (Boxplot) ---
    plt.subplot(1, 3, 3)
    data = [class_conf[c] for c in class_ids]
    plt.boxplot(data, labels=class_names)
    plt.title("Confidence Distribution by Class")
    plt.ylabel("Confidence Score")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("confidence_distributions.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot Inference Time ---
    plt.figure(figsize=(18, 5))
    # 1. Inference time per image
    plt.subplot(1, 3, 1)
    plt.plot(range(len(inference_times)), inference_times, marker='o', color='purple')
    plt.title("Inference Time per Image")
    plt.xlabel("Image Number")
    plt.ylabel("Inference Time (ms)")

    # 2. Inference time distribution
    plt.subplot(1, 3, 2)
    plt.hist(inference_times, bins=15, color='orange', alpha=0.7, edgecolor='black')
    plt.title("Inference Time Distribution")
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Frequency")

    # 3. Inference time statistics
    plt.subplot(1, 3, 3)
    stats = [np.mean(inference_times), np.median(inference_times), np.min(inference_times), np.max(inference_times)]
    labels = ['Mean', 'Median', 'Min', 'Max']
    colors = ['red', 'blue', 'green', 'orange']
    bars = plt.bar(labels, stats, color=colors, alpha=0.7)
    for bar, val in zip(bars, stats):
        plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}", ha='center', va='bottom', fontweight='bold')
    plt.title("Inference Time Statistics")
    plt.ylabel("Time (ms)")

    plt.tight_layout()
    plt.savefig("inference_time_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Print summary ---
    print("\nDetection counts by class:")
    for c in class_ids:
        print(f"{COCO_CLASSES[c]}: {len(class_conf[c])} detections, avg conf={avg_conf[c]:.2f}")

    print("\nInference time stats (ms):")
    print(f"Mean:   {np.mean(inference_times):.2f}")
    print(f"Median: {np.median(inference_times):.2f}")
    print(f"Min:    {np.min(inference_times):.2f}")
    print(f"Max:    {np.max(inference_times):.2f}")

if __name__ == "__main__":
    main()