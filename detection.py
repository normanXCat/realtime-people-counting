from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.predict(source=0, classes=[0], conf=0.5, show=True)

for result in results:
    boxes = result.boxes
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        
        print(f"Personne détectée : x1={int(x1)}, y1={int(y1)}, "
              f"x2={int(x2)}, y2={int(y2)}, confiance={conf:.2f}")
    
    print(f"Nombre de personnes : {len(boxes)}")