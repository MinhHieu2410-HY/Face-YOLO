from ultralytics import YOLO
import torch

#yolo predict model=runs/detect/train/weights/best.pt source=0 imgsz=416 device=0 show_conf=False show=True
#run real time cmd

if __name__ == "__main__":
    # 1. Load model của bạn
    model = YOLO("runs/detect/train/weights/best.pt")

    # 2. Tự nhận GPU nếu có, không thì dùng CPU - chạy được trên mọi máy
    device = 0 if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available()   # half precision chỉ dùng được khi có GPU

    # 3. Chạy predict với cấu hình tiết kiệm VRAM
    # Chạy lệnh này từ thư mục gốc project (nơi có folder Face/), vd: python Test.py
    results = model.predict(
        source="Face/test/images",   # đường dẫn tương đối - ai clone repo cũng chạy được
        imgsz=416,            # Giảm từ 640px xuống 416px
        save=True,
        device=device,
        half=half,             # giảm dung lượng VRAM tiêu thụ (chỉ khi có GPU)
        show_conf=False
    )
