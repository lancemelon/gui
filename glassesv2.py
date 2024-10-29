from ultralytics import YOLO
from collections import defaultdict, deque

# Initialize model
model = YOLO('./best.pt')

# Create a dictionary to track label counts across frames
label_history = defaultdict(lambda: deque(maxlen=3))  # Sliding window of 3 frames per label
locked_labels = set()  # Set of locked labels

# Predict on the video and process frames
res = model.predict('glassesvid.mp4', conf=0.50, save=True, stream=True)

# Iterate over each frame's predictions
for result in res:
    frame_labels = []  # Collect labels from the current frame
    
    # For each detected object, get its class index and map it to the label name
    for cls in result.boxes.cls:
        label = model.names[int(cls)]  # Convert class index to label name
        frame_labels.append(label)

    # Update label history and check for consecutive occurrences
    for label in frame_labels:
        if label not in locked_labels:
            # Add current label to its history buffer (deque of max length 3)
            label_history[label].append(True)

            # Check if label has appeared in 3 consecutive frames
            if len(label_history[label]) == 3 and all(label_history[label]):
                print(f"Label {label} has appeared in 3 consecutive frames. Locking it.")
                locked_labels.add(label)  # Lock the label once it's seen in 3 frames
        else:
            print(f"Label {label} is locked and won't change.")
    
    # Remove labels that are not present in this frame from the history
    for label in list(label_history):
        if label not in frame_labels:
            label_history[label].append(False)
            # If label has not appeared in the last 3 frames, reset its history
            if not any(label_history[label]):
                del label_history[label]

# Print locked labels
print(f"Locked labels: {locked_labels}")
