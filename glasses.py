from ultralytics import YOLO

model = YOLO('best.pt')

res = model.predict('glassesvid.mp4', conf=0.25, save=True, stream=True)
