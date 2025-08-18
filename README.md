<div align="center">

# 🎓 ỨNG DỤNG NHẬN DIỆN CỬ CHỈ TAY HỖ TRỢ <br> GIAO TIẾP CHO NGƯỜI KHUYẾT TẬT

</div>

<div align="center">

<p align="center">
  <img src="images/logo.png" alt="Logo Đại học Đại Nam" width="200"/>
  <img src="images/AIoTLab_logo.png" alt="Logo AIoTLab" width="170"/>
</p>

</div>

<h3 align="center">🔬 Công nghệ AI hỗ trợ giao tiếp cho người khuyết tật thông qua nhận dạng cử chỉ tay</h3>

<p align="center">
  <strong>Hệ thống nhận diện cử chỉ tay tiếng Việt thời gian thực sử dụng Mediapipe và LSTM</strong>
</p>

---

## 🏗️ Kiến trúc hệ thống

<p align="center">
  
  ![image](https://github.com/user-attachments/assets/1144a93e-ac5b-4e27-9446-c1072cb4b44a)
</p>

Hệ thống được thiết kế với kiến trúc ba tầng chính:

1. **📹 Tầng đầu vào**: Thu nhận dữ liệu từ webcam hoặc video, trích xuất 1662 điểm đặc trưng (pose, face, hand) bằng **MediaPipe Holistic**.  
2. **🧠 Tầng mô hình**: Xử lý chuỗi 30 khung hình bằng mạng **LSTM nhiều lớp**.  
3. **🔊 Tầng đầu ra**: Hiển thị nhãn dự đoán trên màn hình và cung cấp phản hồi trực quan/âm thanh nếu độ tin cậy vượt ngưỡng (ví dụ: ≥ 0.8).  

---

## ✨ Tính năng nổi bật

- **Mô hình LSTM** tối ưu cho chuỗi thời gian, đạt độ chính xác cao.  
- **Nhận diện thời gian thực**, với tốc độ xử lý trung bình 20–30 FPS.  
- **Phát hiện hành động ổn định**, hạn chế nhiễu và sai sót khi người dùng thực hiện nhanh.  
- **Phản hồi trực quan và âm thanh**, giúp hỗ trợ giao tiếp hiệu quả.  
- **Nhận diện bộ cử chỉ tiếng Việt** gồm: “Xin chào”, “Cảm ơn”, “Xin lỗi”, “Tạm biệt”, “Hạnh phúc”, “Tuyệt vời”, “Yêu thương”, “Biết ơn”, “Ghét”, cùng nhãn “null”.  

---

## 🔧 Công nghệ sử dụng

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/)  
[![Mediapipe](https://img.shields.io/badge/Mediapipe-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)  

---

## 📥 Cài đặt

### 🛠️ Yêu cầu hệ thống
- **Python** `3.8+`  
- **Webcam** (khuyến nghị 1280x720)  
- **RAM** `4GB+`  
- **CPU** `2+ nhân`  
- **Dung lượng lưu trữ** `2GB+`  

### ⚙️ Hướng dẫn cài đặt
1. **Tải mã nguồn**
   ```bash
   git clone https://github.com/ten-cua-ban/gesture-recognition
   cd gesture-recognition
