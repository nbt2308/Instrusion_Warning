from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
from ultralytics import YOLO
from telegram_utils import send_telegram
import datetime
import threading
import torch
import pygame
from face_utils import FaceRecognizer


# Hàm check xem điểm centroid của object có nằm trong polygon hay không  
def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    print(polygon.contains(centroid))
    return polygon.contains(centroid)

#Nhận diện
class YoloDetect:
    def __init__(self, model_path="yolo11n.pt"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(device)
        self.device = device
        print(f"YOLO model is running on: {self.device}")
        self.last_alert = None
        self.alert_interval = 5 # giây
        self.face_recognizer = FaceRecognizer("authorized_face") 
        

    # Hàm tính toán và vẽ tâm centroid
    def draw_centroid(self, frame, corners, draw=True, color=(0, 255, 0)):
        pt1, pt2, pt3, pt4 = corners
        #hàm vẽ đường chéo để tìm centroid. Xem hướng dẫn trong thư mục "Tutorial"
        def line_intersection(p1, p2, p3, p4): #Ax + By= C
            # Đường chéo thứ nhất
            A1 = p2[1] - p1[1]
            B1 = p1[0] - p2[0]
            C1 = A1 * p1[0] + B1 * p1[1]
            # Đường chéo thứ hai
            A2 = p4[1] - p3[1]
            B2 = p3[0] - p4[0]
            C2 = A2 * p3[0] + B2 * p3[1]

            #Biến det là để check xem 2 đường chéo có song song hoặc trùng nhau hay không
            det = A1 * B2 - A2 * B1
            #Nếu song song hoặc trùng nhau thì trả về none(tức là không có điểm centroid)
            if det == 0:
                return None
            #Nếu không song song hoặc trùng nhau thì tính toán theo công thức này để trả về toạ độ(x,y) của tâm centroid
            else:
                x = (B2 * C1 - B1 * C2) / det
                y = (A1 * C2 - A2 * C1) / det
                return int(x), int(y)
        # Vẽ centroid truyền tham số là các đỉnh(góc của object) vào.
        # Vào thư mục "Tutorial" để xem tại sao lại truyền theo thứ tự (pt1, pt3, pt2, pt4)
        centroid = line_intersection(pt1, pt3, pt2, pt4)

        #hiện tâm centroid
        if draw and centroid:
            # Hiển thị tâm centroid màu đỏ, bán kính=5, thickness=-1(Tô kín)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            """
            ### Hai lệnh này dùng để hiển thị 2 đường chéo của object(Nếu muốn)
            cv2.line(img, pt1, pt3, color, 1) 
            cv2.line(img, pt2, pt4, color, 1)
            
            ### Lệnh này dùng để hiển thị chữ "Centroid" màu đỏ ngay vị trí toạ độ centroid.x + 5 và centroid.y - 5(tức là ngay kế bên centroid)
            cv2.putText(img, "Centroid", (centroid[0] + 5, centroid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            """

        return centroid


    def detect(self, frame,points):
        # Sử dụng model của YOLO để detect
        results =  self.model(frame)
        
        """
        ### result[0] là một list các kết quả tương ứng với các ảnh đã xử lý
        ### Bên trong results[0] có thuộc tính .boxes, chứa danh sách các object (bounding boxes) mà YOLO phát hiện được trên frame đó, trong trường hợp này là 0 : 'person'
        ### Mỗi box là 1 object( ví dụ như: {0: con người(person), 60: cái ghế(chair),...})
        """
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            class_id = int(box.cls[0])  # Lấy class ID, cls[0]: lấy ra id của person
            class_name = self.model.names[class_id]  # Lấy class_name từ class_id
            confidence = float(box.conf[0]) # Lấy confidence score(độ tin cậy) mà YOLO dự đoán cho object đó
            # Chỉ nhận diện class person
            if class_name != "person":
                continue
            # Vẽ bounding box và hiển thị label của object
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Tính 4 góc từ bbox
            pt1 = (x1, y1) # top-left
            pt2 = (x2, y1) # top-right
            pt3 = (x2, y2) # bottom-right
            pt4 = (x1, y2) # bottom-left

            # Vẽ centroid lên frame khi đã detect được object
            centroid=self.draw_centroid(frame, [pt1, pt2, pt3, pt4])
            # Check xem có tồn tại centroid hay centroid có nằm trong polygon hay không
            # Check xem object có trong vùng cấm không
            bbox = (x1, y1, x2, y2)
            authorized = self.face_recognizer.is_authorized(frame, bbox)

            # Luôn dò khuôn mặt để vẽ label & bbox
            person_crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            faces = self.face_recognizer.face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                face_x1, face_y1 = x1 + fx, y1 + fy
                face_x2, face_y2 = face_x1 + fw, face_y1 + fh

                # Vẽ khung khuôn mặt
                color = (0, 255, 0) if authorized else (0, 0, 255)
                label = "Authorized" if authorized else "Unauthorized"
                cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), color, 2)
                cv2.putText(frame, label, (face_x1, face_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Cảnh báo nếu là người lạ và centroid nằm trong vùng cấm
            if not authorized and centroid and isInside(points, centroid):
                self.alert(frame)

            # if centroid and isInside(points, centroid):  
            #   self.alert(frame)

        return frame

    def alert(self, frame):
        cv2.putText(frame, "ALERTTTT!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Đo khoảng thời gian hiện tại theo chuẩn UTC(quốc tế)
        now = datetime.datetime.utcnow()
        # Tính khoảng thời gian từ lần gửi cảnh báo trước đến hiện tại
        if self.last_alert is None or (now - self.last_alert).total_seconds() >= self.alert_interval:
            self.last_alert = now
            # Khi có người xâm nhập sẽ phát ra âm thanh cảnh báo
            try:
                pygame.mixer.init()
                pygame.mixer.music.load("sounds/alert.mp3")
                pygame.mixer.music.play()
            except Exception as e:
                print("Lỗi âm thanh:", e)
            # Lưu ảnh cảnh báo(ảnh này sẽ được cập nhật sau mỗi lần cảnh báo)
            cv2.imwrite("Alert_nofications/alert.png", frame)

            # Gửi qua Telegram bằng thread
            thread = threading.Thread(target=send_telegram)
            thread.start()