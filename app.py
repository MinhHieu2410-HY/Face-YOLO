from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from yolo_utils import (
    load_model, predict_frame, class_counts, process_video,
    dataset_summary, load_data_yaml, dataset_root, extract_zip,
    collect_pairs_from_dir, validate_pairs, merge_pairs,
)

CFG = {
    "weights_best": "runs/detect/train/weights/best.pt",
    "weights_last": "runs/detect/train/weights/last.pt",
    "data_yaml": "Face/data.yaml",
    "imgsz": 416,
    "conf": 0.25,
    "iou": 0.45,
    "device": 0,
    "half": True,
}

st.set_page_config(page_title="Face-YOLO Demo", layout="wide")


@st.cache_resource
def get_model(weights_path):
    return load_model(weights_path)


def sidebar_config():
    st.sidebar.header("Cấu hình mô hình")
    weights_choice = st.sidebar.radio("Trọng số", ["best.pt", "last.pt", "Khác..."])
    if weights_choice == "best.pt":
        weights_path = st.sidebar.text_input("Đường dẫn best.pt", value=CFG["weights_best"])
    elif weights_choice == "last.pt":
        weights_path = st.sidebar.text_input("Đường dẫn last.pt", value=CFG["weights_last"])
    else:
        weights_path = st.sidebar.text_input("Đường dẫn weights tùy chỉnh", value="")

    imgsz = st.sidebar.select_slider("Kích thước ảnh (imgsz)", options=[320, 416, 512, 640], value=CFG["imgsz"])
    conf = st.sidebar.slider("Ngưỡng tin cậy (conf)", 0.05, 0.95, CFG["conf"], 0.05)
    iou = st.sidebar.slider("Ngưỡng IoU (NMS)", 0.1, 0.9, CFG["iou"], 0.05)
    device_options = [0, "cpu"] if torch.cuda.is_available() else ["cpu"]
    device = st.sidebar.selectbox("Thiết bị", device_options, format_func=lambda x: "GPU (0)" if x == 0 else "CPU")
    half = st.sidebar.checkbox("Half precision (FP16, chỉ dùng khi có GPU)", value=CFG["half"])
    if device == "cpu":
        half = False

    return weights_path, imgsz, conf, iou, device, half


def image_tab(model, imgsz, conf, iou, device, half):
    uploaded = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "bmp"], key="img_uploader")
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated, r = predict_frame(model, frame_bgr, imgsz, conf, iou, device, half)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Kết quả nhận diện", use_container_width=True)

    counts = class_counts(r, model)
    if counts:
        st.write("Số lượng theo lớp:", counts)
    else:
        st.info("Không phát hiện đối tượng nào.")


def video_tab(model, imgsz, conf, iou, device, half):
    uploaded = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    if uploaded is None:
        return

    in_path = str(Path(tempfile.gettempdir()) / f"cardyolo_in_{uploaded.name}")
    out_path = str(Path(tempfile.gettempdir()) / "cardyolo_out.mp4")

    if st.button("Xử lý video", key="process_video_btn"):
        Path(in_path).write_bytes(uploaded.getvalue())
        progress = st.progress(0.0)
        preview = st.empty()
        process_video(
            model, in_path, out_path, imgsz, conf, iou, device, half,
            progress_cb=progress.progress,
            preview_cb=lambda f: preview.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)),
        )
        st.session_state["video_out_path"] = out_path
        st.success("Xử lý xong.")

    if st.session_state.get("video_out_path"):
        st.video(st.session_state["video_out_path"])
        with open(st.session_state["video_out_path"], "rb") as f:
            st.download_button("Tải video kết quả", f, file_name="ket_qua_nhan_dien.mp4")


def webcam_tab(model, imgsz, conf, iou, device, half):
    photo = st.camera_input("Chụp ảnh từ camera")
    if photo is None:
        return

    image = Image.open(photo).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated, r = predict_frame(model, frame_bgr, imgsz, conf, iou, device, half)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Kết quả nhận diện", use_container_width=True)

    counts = class_counts(r, model)
    if counts:
        st.write("Số lượng theo lớp:", counts)
    else:
        st.info("Không phát hiện đối tượng nào.")


def test_tab(model, imgsz, conf, iou, device, half):
    mode = st.radio("Loại đầu vào", ["Ảnh", "Video", "Camera"], horizontal=True)
    st.divider()
    if mode == "Ảnh":
        image_tab(model, imgsz, conf, iou, device, half)
    elif mode == "Video":
        video_tab(model, imgsz, conf, iou, device, half)
    else:
        webcam_tab(model, imgsz, conf, iou, device, half)


def data_train_tab(device):
    data_yaml_path = st.text_input("Đường dẫn data.yaml", value=CFG["data_yaml"])
    if not Path(data_yaml_path).exists():
        st.warning(f"Không tìm thấy data.yaml tại: {data_yaml_path}")
        return

    st.subheader("Thông tin dataset hiện tại")
    st.json(dataset_summary(data_yaml_path))

    st.subheader("Thêm dữ liệu bên ngoài")
    st.caption(
        "Tải lên 1 file .zip chứa ảnh (.jpg/.jpeg/.png) và label YOLO (.txt) cùng tên file, "
        "ví dụ img001.jpg đi kèm img001.txt. Mỗi dòng label: class_id x_center y_center width height (đã chuẩn hóa 0-1)."
    )
    zip_file = st.file_uploader("File .zip dữ liệu mới", type=["zip"], key="data_zip_uploader")
    split = st.radio("Thêm vào tập", ["train", "val"], horizontal=True)

    if zip_file is not None and st.button("Kiểm tra & thêm dữ liệu"):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "upload.zip"
            zip_path.write_bytes(zip_file.getvalue())
            extracted = extract_zip(str(zip_path), Path(tmpdir) / "extracted")
            image_files, label_files = collect_pairs_from_dir(extracted)

            if not image_files:
                st.error("Không tìm thấy ảnh nào trong file zip.")
            else:
                data_cfg = load_data_yaml(data_yaml_path)
                nc = data_cfg.get("nc", 0)
                valid_pairs, errors = validate_pairs(image_files, label_files, nc)

                if errors:
                    st.error(f"Có {len(errors)} lỗi ở các file sau:")
                    st.code("\n".join(errors[:50]))
                    if len(errors) > 50:
                        st.caption(f"... và {len(errors) - 50} lỗi khác (đã ẩn bớt).")

                if valid_pairs:
                    root = dataset_root(data_yaml_path, data_cfg)
                    added = merge_pairs(valid_pairs, root, data_cfg, split)
                    st.success(f"Đã thêm {added} cặp ảnh/label hợp lệ vào tập '{split}'.")

    st.divider()
    st.subheader("Huấn luyện lại")
    base_choice = st.selectbox("Bắt đầu từ", ["best.pt hiện tại", "last.pt hiện tại", "yolov8n.pt (huấn luyện mới)"])
    epochs = st.number_input("Số epochs", min_value=1, value=10, step=1)
    imgsz_train = st.select_slider("Kích thước ảnh huấn luyện", options=[320, 416, 512, 640], value=640)
    run_name = st.text_input("Tên lần chạy (run name)", value="train_demo")

    if st.button("Bắt đầu huấn luyện"):
        base_map = {
            "best.pt hiện tại": CFG["weights_best"],
            "last.pt hiện tại": CFG["weights_last"],
            "yolov8n.pt (huấn luyện mới)": "yolov8n.pt",
        }
        base_path = base_map[base_choice]
        with st.spinner("Đang huấn luyện, vui lòng đợi..."):
            train_model = load_model(base_path)
            train_model.train(
                data=data_yaml_path, epochs=int(epochs), imgsz=imgsz_train,
                device=device, project="runs/detect", name=run_name,
            )
            run_dir = Path(train_model.trainer.save_dir)

        st.success(f"Huấn luyện xong. Kết quả lưu tại: {run_dir}")

        results_png = run_dir / "results.png"
        if results_png.exists():
            st.image(str(results_png), caption="Biểu đồ kết quả huấn luyện")

        best_after = run_dir / "weights" / "best.pt"
        if best_after.exists():
            st.info(f"Trọng số mới tốt nhất: {best_after}")


def main():
    st.title("Face-YOLO — Demo phát hiện khuôn mặt")
    st.caption("Nhận diện khuôn mặt bằng YOLOv8: kiểm tra qua ảnh/video/camera, và thêm dữ liệu huấn luyện mới.")

    weights_path, imgsz, conf, iou, device, half = sidebar_config()

    if not weights_path or not Path(weights_path).exists():
        st.warning(f"Không tìm thấy file weights tại: {weights_path or '(chưa nhập)'}")
        st.stop()

    try:
        model = get_model(weights_path)
    except Exception as e:
        st.error(f"Không tải được mô hình: {e}")
        st.stop()

    tab_test, tab_data = st.tabs(["Kiểm tra (Test)", "Dữ liệu & Huấn luyện (Train)"])
    with tab_test:
        test_tab(model, imgsz, conf, iou, device, half)
    with tab_data:
        data_train_tab(device)


if __name__ == "__main__":
    main()
