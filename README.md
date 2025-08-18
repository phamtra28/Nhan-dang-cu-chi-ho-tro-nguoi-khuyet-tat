<div align="center">

# 🎓 NHẬN DIỆN CỬ CHỈ HỖ TRỢ NGƯỜI KHUYẾT TẬT  

<img src="anh/logo.jpg" alt="Logo" width="1000"/>

--- 

### 🔬 Ứng dụng trí tuệ nhân tạo trong giao tiếp hỗ trợ người khuyết tật thông qua nhận dạng cử chỉ 

**Hệ thống nhận diện ngôn ngữ ký hiệu tiếng Việt thời gian thực sử dụng Mediapipe và LSTM**  

</div>

---

## 🔎 Giới thiệu  

Người khiếm thính và khiếm ngôn gặp rất nhiều khó khăn trong quá trình giao tiếp hằng ngày. Ngôn ngữ ký hiệu là phương tiện quan trọng giúp họ truyền đạt thông tin, nhưng lại không phải ai trong cộng đồng cũng có khả năng hiểu và sử dụng ngôn ngữ này. Do đó, việc phát triển một hệ thống có thể tự động **nhận diện cử chỉ và chuyển đổi thành văn bản hoặc giọng nói** là một hướng đi cần thiết, góp phần **thu hẹp khoảng cách giao tiếp** và **nâng cao chất lượng cuộc sống** cho người khuyết tật.  

---

## 🏗️ Kiến trúc hệ thống  

Quy trình hoạt động của hệ thống nhận dạng cử chỉ tay được triển khai qua các bước sau:  

1. **Thu thập dữ liệu**: Sử dụng tập dữ liệu **QiPedC**.  
2. **Trích xuất keypoints**: Áp dụng **MediaPipe Holistic** để lấy ra **1662 điểm đặc trưng** (pose, face, hands).  
3. **Tiền xử lý dữ liệu**: Chuẩn hóa, padding chuỗi 30 khung hình và lưu dưới dạng **`.npy`**.  
4. **Huấn luyện mô hình**: Sử dụng mạng **LSTM nhiều lớp (128 → 256 → 128)**, sau đó thêm lớp **Dense 64** và **Softmax** để phân loại cử chỉ.  
5. **Dự đoán thời gian thực**: Lấy dữ liệu trực tiếp từ camera, xử lý theo chuỗi 30 khung hình.  
6. **Hiển thị kết quả**: Xuất ra **nhãn cử chỉ** kèm **độ tin cậy (confidence score)** ngay trên giao diện.  

---

## ✨ Tính năng chính  

- **Nhận diện cử chỉ tay thời gian thực** trực tiếp từ camera với tốc độ trung bình 20–30 FPS.  
- **Trích xuất keypoints tự động** bằng MediaPipe Holistic, gồm 1662 điểm đặc trưng từ khuôn mặt, bàn tay và dáng người.  
- **Huấn luyện và dự đoán bằng LSTM** nhiều lớp, đảm bảo khả năng học chuỗi động tác và phân loại chính xác cử chỉ.  
- **Hiển thị kết quả trực quan** ngay trên giao diện: gồm nhãn cử chỉ và độ tin cậy (confidence score).  
- **Khả năng mở rộng**: Dễ dàng bổ sung thêm hành động mới bằng cách thu thập dữ liệu và huấn luyện lại mô hình.  

---

## 🔧 Công nghệ sử dụng  

- [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=yellow)](https://www.python.org/)  
- [![Mediapipe](https://img.shields.io/badge/Mediapipe-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev/)  
- [![LSTM](https://img.shields.io/badge/LSTM-FF6F00?style=for-the-badge&logo=keras&logoColor=white)](https://en.wikipedia.org/wiki/Long_short-term_memory)  
- [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)  

---

## 📊 Kết quả thử nghiệm  

- **Độ chính xác**: trung bình đạt **90%** trên tập kiểm tra.  
- **Thời gian suy luận**: khoảng **30ms mỗi khung hình**, đủ nhanh cho nhận diện thời gian thực.  
- **Tốc độ hoạt động**: duy trì **20–33 FPS** trên webcam chuẩn 1280x720.  
- **Các lỗi thường gặp**: nhầm lẫn giữa cử chỉ tương tự, hoặc khi ánh sáng yếu.  
- **Nhận xét**: hệ thống đã chứng minh tính khả thi và có tiềm năng triển khai thực tế để hỗ trợ cộng đồng người khuyết tật.  

---

## ⚠️ Hạn chế và hướng phát triển  

- **Hạn chế**:  
  - Chỉ nhận diện được các cử chỉ đã huấn luyện.  
  - Độ chính xác giảm khi môi trường ánh sáng không tốt hoặc cử chỉ được thực hiện quá nhanh.  
  - Chưa hỗ trợ bộ dữ liệu ngôn ngữ ký hiệu đầy đủ.  

- **Hướng phát triển**:  
  - **Mở rộng bộ dữ liệu**: thu thập thêm dữ liệu từ nhiều người, nhiều góc quay, đa dạng điều kiện môi trường.  
  - **Tối ưu mô hình**: áp dụng *pruning*, *quantization* để tăng tốc độ trên thiết bị di động.  
  - **Tích hợp giọng nói tự nhiên** thay vì phát âm thanh thô để giao tiếp gần gũi hơn.  
  - **Ứng dụng di động**: xây dựng phiên bản chạy trực tiếp trên smartphone, giúp người khuyết tật sử dụng mọi lúc mọi nơi.  

---

<div align="center">

📝  © 2025 – Phạm Văn Trà, Nhóm 12 - CNTT 17-05, Khoa Công nghệ Thông tin, Trường Đại học Đại Nam.  
👩‍🏫 **GV hướng dẫn**: Lê Trung Hiếu, Nguyễn Thái Khánh  

</div>
