<div align="center">

# 🎓 ỨNG DỤNG NHẬN DIỆN CỬ CHỈ TAY HỖ TRỢ GIAO TIẾP CHO NGƯỜI KHUYẾT TẬT  

<img src="images/logo.png" alt="Logo Đại học Đại Nam" width="200"/>
<img src="images/AIoTLab_logo.png" alt="Logo AIoTLab" width="170"/>

---

### 🔬 Công nghệ AI hỗ trợ giao tiếp cho người khuyết tật qua nhận dạng cử chỉ tay  

**Hệ thống nhận diện cử chỉ tay tiếng Việt thời gian thực sử dụng Mediapipe và LSTM**  

</div>

---

Hệ thống được thiết kế với **kiến trúc ba tầng**: (1) thu video từ webcam và trích xuất 1662 điểm đặc trưng bằng *MediaPipe Holistic*; (2) xử lý chuỗi 30 khung hình liên tiếp bằng *mạng LSTM nhiều lớp*; (3) hiển thị nhãn dự đoán và độ tin cậy, đồng thời phát âm thanh hỗ trợ giao tiếp.  

Hệ thống có khả năng **nhận diện thời gian thực** với tốc độ 20–30 FPS, độ chính xác trung bình 85–90% trên tập kiểm tra, và hỗ trợ **10 cử chỉ tiếng Việt**: “null”, “xin chào”, “cảm ơn”, “xin lỗi”, “tạm biệt”, “hạnh phúc”, “tuyệt vời”, “yêu thương”, “biết ơn”, “ghét”.  

Quy trình triển khai bao gồm các bước:  
- **Thu thập dữ liệu** từ camera và lưu keypoints dưới dạng `.npy`.  
- **Xử lý và chia tập dữ liệu** thành train/validation/test.  
- **Huấn luyện mô hình LSTM** và lưu phiên bản tốt nhất dưới dạng `best_model.keras`.  
- **Dự đoán thời gian thực** với webcam, hiển thị nhãn dự đoán và phát âm thanh khi độ tin cậy vượt ngưỡng 0.8.  

Kết quả thử nghiệm cho thấy hệ thống hoạt động ổn định, có thể phân biệt các cử chỉ cơ bản và phản hồi ngay lập tức. Tuy nhiên, vẫn tồn tại hạn chế khi điều kiện ánh sáng không tốt hoặc khi các cử chỉ có hình dạng tương tự nhau, dẫn đến nhầm lẫn (ví dụ: “xin chào” ↔ “cảm ơn”).  

Trong tương lai, hệ thống có thể được **mở rộng bộ dữ liệu**, **đa dạng hóa điều kiện thu thập**, và **tối ưu mô hình** (pruning, quantization) để triển khai trên các thiết bị cấu hình thấp. Ngoài ra, có thể tích hợp thêm **giọng nói tự nhiên** nhằm nâng cao trải nghiệm giao tiếp cho người khuyết tật.  

---

<div align="center">

📝 **Bản quyền**: © 2025 – Phạm Văn Trà, Nhóm CNTT, Khoa Công nghệ Thông tin, Đại học Đại Nam.  
👩‍🏫 **GV hướng dẫn**: Lê Diệu Anh.  

</div>
