import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Tải mô hình đã được huấn luyện và danh sách các cử chỉ
try:
    model = load_model('D:/HọcTập/Nam3-Ki1/New folder/BTL-CHUYENDOISO/best_model.keras')
    action_list = np.load('gesture_data/action_map.npy', allow_pickle=True)
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy 'final_model.keras' hoặc 'gesture_data/action_map.npy'.")
    print("Vui lòng đảm bảo bạn đã chạy thành công đoạn mã huấn luyện trước đó.")
    exit()

# Cấu hình MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Xử lý ảnh để phát hiện các điểm mốc của tư thế, khuôn mặt và tay."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """Trích xuất một mảng phẳng các keypoints từ kết quả của MediaPipe."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    """Vẽ các điểm mốc với màu sắc và kiểu dáng tùy chỉnh."""
    # Vẽ các kết nối của pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    # Vẽ các kết nối của khuôn mặt
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    # Vẽ các kết nối của tay trái
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    # Vẽ các kết nối của tay phải
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

def main():
    """Hàm chính để chạy chương trình nhận diện cử chỉ theo thời gian thực."""
    sequence = []
    sentence = []
    predictions = []
    
    # Bạn có thể thay đổi giá trị này để điều chỉnh độ nhạy
    threshold = 0.1 
    
    cooldown_frames = 15 
    cooldown_counter = 0
    SEQUENCE_LENGTH = 30 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Lỗi: Không thể mở webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("💡 Bắt đầu nhận diện cử chỉ theo thời gian thực. Nhấn 'q' để thoát.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:] 

            if len(sequence) == SEQUENCE_LENGTH:
                print(f"Chuỗi đủ {SEQUENCE_LENGTH} khung hình. Đang dự đoán...")
                
                if cooldown_counter == 0:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    
                    print("Các dự đoán:", res)
                    print("Dự đoán cao nhất:", action_list[np.argmax(res)], f"với độ tin cậy {res[np.argmax(res)]:.2f}")

                    if res[np.argmax(res)] > threshold:
                        predicted_action = action_list[np.argmax(res)]
                        confidence = res[np.argmax(res)]
                        
                        if len(sentence) > 0 and predicted_action != sentence[-1]:
                            sentence = [predicted_action]
                        elif not sentence:
                            sentence = [predicted_action]
                        
                        cooldown_counter = cooldown_frames
                    
                else:
                    cooldown_counter -= 1
            
            display_text = ""
            confidence_text = ""
            if len(sequence) == SEQUENCE_LENGTH:
                # Lấy dự đoán mới nhất
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                confidence = res[np.argmax(res)]
                
                display_text = f"Cu chi: {action_list[np.argmax(res)]}"
                confidence_text = f"Do tin cay: {confidence:.2f}"
            else:
                display_text = "Dang thu thap du lieu..."
                
            cv2.putText(image, display_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, confidence_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Nhan dien cu chi', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Đã thoát chương trình. Tạm biệt!")

if __name__ == '__main__':
    main()
