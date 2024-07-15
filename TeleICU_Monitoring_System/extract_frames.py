import cv2
import os
import shutil
import numpy as np  # Import numpy

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define specific class names
desired_classes = ['nurse', 'intensivist', 'family_member', 'patient', 'others']

# Map COCO indices to desired classes
class_indices = {
    classes.index('person'): 'nurse',  # Assuming 'person' class covers nurse, doctor, patient, etc.
    # You can add more mappings as needed based on your specific classes
}

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in class_indices:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append((x, y, w, h, confidence, class_indices[class_id]))

    return boxes

def extract_frames(video_path, output_dir, split_ratio=0.8):
    # Create directories for training and validation datasets
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    classes = ['nurse', 'intensivist', 'family_member', 'patient', 'others']

    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1

    cap.release()

    # Calculate split index based on split_ratio
    split_idx = int(count * split_ratio)

    # Save frames to respective directories
    for idx, frame in enumerate(frames):
        if idx < split_idx:
            output_folder = train_dir
        else:
            output_folder = val_dir
        
        # Detect objects and draw bounding boxes
        boxes = detect_objects(frame)
        for (x, y, w, h, conf, obj_class) in boxes:
            label = f"{obj_class} {conf:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save frame to directory based on the detected class
            frame_path = os.path.join(output_folder, obj_class, f'frame_{count:04d}.jpg')
            cv2.imwrite(frame_path, frame.copy())  # Use frame.copy() to avoid overwriting

    print(f'Extracted {count} frames from {video_path} and organized them into training and validation sets.')
    print('Success')

# Example usage
video_path = 'vid.mp4'
output_dir = 'dataset'
extract_frames(video_path, output_dir)
