from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO("./best.pt")  # Adjust the path if needed

# Dictionaries to track detections
seen = defaultdict(int)  # To count how many times each label has been seen
conf = defaultdict(float)  # To accumulate confidence scores

game = []  # List to store the top labels

# Make predictions
results = model.predict("glassesTest\glassesvid.mp4", conf=0.50, save=True, stream=True)

for result in results:
    frame_labels = defaultdict(float)  # Store confidence per label for the current frame

    for bbox in result.boxes:
        label = model.names[int(bbox.cls)]  # Get the label name
        confidence = bbox.conf[0]  # Get the raw confidence score
        frame_labels[label] += confidence  # Accumulate confidence for this frame

    for label in frame_labels:
        seen[label] += 1  # Count how many times the label has been seen
        conf[label] += frame_labels[label]  # Accumulate total confidence

# Calculate average confidence as a percentage
for label in conf:
    if seen[label] > 0:
        conf[label] = (conf[label] / seen[label]) * 100  # Average confidence

# Get the top 7 labels based on average confidence
top_n = 7  # Number of top labels to retrieve
for _ in range(top_n):
    if conf:  # Ensure there are still labels left to process
        max_conf = max(conf, key=conf.get)  # Find the label with the maximum average confidence
        game.append(max_conf)  # Add it to the game state
        del conf[max_conf]  # Remove it from the dictionary to avoid duplicates

print(f"State of the Game: {game}")
