from ultralytics import YOLO

model = YOLO('yolov8x')

results = model.predict('input_videos/08fd33_4.mp4',save=True) #Your video here
print(results[0]) #first frame results

print('=====================================')

for box in results[0].boxes:
    print(box)