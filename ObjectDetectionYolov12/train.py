from ultralytics import YOLO

def main():
    # Path to your model and dataset config
    model_path = "yolov12n.pt"
    data_yaml = "coco128.yaml"  # Make sure this points to your coco128.yaml

    # Create and train the model
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0  # set to 'cpu' if you don't have a GPU
    )

    # Save the trained model
    model.save("yolov12s_trained.pt")

if __name__ == "__main__":
    main()