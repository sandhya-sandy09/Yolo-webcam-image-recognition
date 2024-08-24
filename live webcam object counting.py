##counting objects

import cv2
import supervision as sv
from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO('yolov8l.pt')

    # Set up webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load class names from the model
    class_names = model.names  # Assuming this is how you get the class names

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference
        results = model(frame)
        
        # Extract detections from results
        detections = results[0].boxes  # Access the detection boxes
        object_counts = {}  # Dictionary to store object counts

        if detections is not None:
            for detection in detections.data:
                # Extract coordinates, confidence, and class ID
                x1, y1, x2, y2, confidence, class_id = detection
                
                # Convert coordinates to integer
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Get class name
                class_name = class_names[int(class_id)]
                
                # Count the number of objects for each class
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the counts on the frame
        y_offset = 20
        for class_name, count in object_counts.items():
            count_text = f"{class_name}: {count}"
            cv2.putText(frame, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 30

        # Display the frame
        cv2.imshow("YOLOv8 Live", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

