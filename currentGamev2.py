# More Pythonic Format for currentGame.py
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO("./best.pt")  # Adjust the path if needed

# Dictionaries to track detections
seen = defaultdict(int)  # To count how many times each label has been seen
conf = defaultdict(float)  # To accumulate confidence scores
game = []  # List to store the top labels

# Make predictions
results = model.predict("glassesTest/glassesvid.mp4", conf=0.50, save=True, stream=True)

def process_frame(frame):
    """Process each frame and update confidence and seen counts."""
    frame_labels = defaultdict(float)  # Store confidence per label for the current frame

    for bbox in frame.boxes:
        label = model.names[int(bbox.cls)]  # Get the label name
        confidence = bbox.conf[0]  # Get the raw confidence score
        frame_labels[label] += confidence  # Accumulate confidence for this frame

    for label, confidence in frame_labels.items():
        seen[label] += 1  # Count how many times the label has been seen
        conf[label] += confidence  # Accumulate total confidence

# Process each result
for result in results:
    process_frame(result)

# Calculate average confidence as a percentage
conf = {label: (score / seen[label]) * 100 for label, score in conf.items() if seen[label] > 0}

# Get the top 7 labels based on average confidence
top_n = 7  # Number of top labels to retrieve
game = [max(conf, key=conf.get) for _ in range(min(top_n, len(conf))) if conf and (conf := {k: conf[k] for k in conf if k != max(conf, key=conf.get)})]

print(f"State of the Game: {game}")
