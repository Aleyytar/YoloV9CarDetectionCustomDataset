from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict("example.png" , imgsz=640 , conf=0.3, save = True, show=True)
