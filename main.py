import cv2
import numpy as np
from yolodetect import YoloDetect

#Sử dụng webcam của thiết bị(mặc định là 0)
cap = cv2.VideoCapture(2)

# Chứa các điểm người dùng click để tạo đa giác
points = []

# Sử dụng các model YOLO
model = YoloDetect("model/yolo11n.pt")

# lưu các điểm click chuột trái đó vào list points
def handle_left_click(event, x, y,flags,points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


# Dùng để vẽ polygon vào frame bằng các điểm được lưu trong list points
def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame


detect = False
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Vẽ polygon
    frame = draw_polygon(frame, points)

    if detect:
        result = model.detect(frame,points)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        points.append(points[0])
        detect = True

    cv2.putText(frame,"Press d to detect", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (247, 0, 0), 2)
    cv2.putText(frame,"Press q to quit", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (247, 0, 0), 2)
    # Hien anh ra man hinh
    cv2.imshow("Intrusion Warning", frame)

    cv2.setMouseCallback('Intrusion Warning', handle_left_click, points)

cap.release()
cv2.destroyAllWindows()