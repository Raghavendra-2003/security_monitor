from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define target street object classes (using COCO dataset class names)
# You can find the full list of COCO classes by printing model.names
target_class_ids = [0, 2, 3, 5, 7, 8, 9, 10, 11] # person, car, motorcycle, bus, truck, traffic light, fire hydrant, stop sign, parking meter
target_class_names = {
    0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
    9: 'traffic light', 11: 'stop sign' # Using a subset for demonstration
}

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame with confidence threshold
    results = model(frame, conf=0.4, verbose=False) # verbose=False suppresses logging for cleaner output

    # Get the first result object (assuming batch size of 1)
    # The plot() method automatically draws boxes and labels
    annotated_frame = results[0].plot()

    # Initialize a dictionary to store object counts for this frame
    object_counts = {name: 0 for name in target_class_names.values()}

    # Iterate through detections and count target objects
    for r in results[0].boxes:
        class_id = int(r.cls[0]) # Get the class ID
        if class_id in target_class_ids:
            class_name = model.names[class_id] # Get the class name from the model's mapping
            if class_name in object_counts: # Ensure it's one of our target names
                object_counts[class_name] += 1

    # Display counts on the annotated frame
    y_offset = 30
    for name, count in object_counts.items():
        text = f"{name}: {count}"
        cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    # Display the frame
    cv2.imshow('Smart City Street Monitor', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()