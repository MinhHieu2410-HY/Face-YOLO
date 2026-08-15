from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # 1. Tải mô hình YOLOv8 kích thước nano (phù hợp máy cấu hình vừa phải)
    model = YOLO("yolov8n.pt")

    # 2. Tự nhận GPU nếu có, không thì dùng CPU - chạy được trên mọi máy
    device = 0 if torch.cuda.is_available() else "cpu"

    # 3. Tiến hành huấn luyện với data của bạn
    # Chạy lệnh này từ thư mục gốc project (nơi có folder Face/), vd: python Train.py
    results = model.train(
        data="Face/data.yaml",   # đường dẫn tương đối - ai clone repo cũng chạy được
        epochs=3,                 # số lượt huấn luyện (thử nghiệm trước với 10, 50 hoặc 100)
        imgsz=640,                 # kích thước ảnh đầu vào (chuẩn là 640)
        device=device              # tự động chọn GPU nếu có, không thì CPU
    )
