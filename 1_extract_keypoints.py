import cv2
import os
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Cấu trúc thư mục:
# gesture_data/
# ├── raw_videos/
# │   ├── gesture1/
# │   │   ├── video1.mp4
# │   │   ├── video2.mp4
# │   │   └── ...
# │   ├── gesture2/
# │   │   ├── video1.mp4
# │   │   ├── video2.mp4
# │   │   └── ...
# │   └── ...
# └── keypoints/

VIDEO_ROOT_DIR = 'gesture_data/raw_videos'
OUTPUT_DIR = 'gesture_data/keypoints'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_video(video_path):
    """Trích xuất keypoints từ một video"""
    holistic = mp_holistic.Holistic(
        static_image_mode=False, 
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    keypoints_all_frames = []
    frame_count = 0
    max_frames = 60  # Giới hạn số frame để xử lý (tùy chọn)

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Xử lý frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        
        # Tạo keypoints cho frame hiện tại
        keypoints = []
        
        # Pose (33 landmarks x 4: x,y,z,visibility)
        if results.pose_landmarks:
            keypoints.extend(np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            ).flatten())
        else:
            keypoints.extend(np.zeros(33 * 4))

        # Face (468 x 3)
        if results.face_landmarks:
            keypoints.extend(np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
            ).flatten())
        else:
            keypoints.extend(np.zeros(468 * 3))

        # Left hand (21 x 3)
        if results.left_hand_landmarks:
            keypoints.extend(np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).flatten())
        else:
            keypoints.extend(np.zeros(21 * 3))

        # Right hand (21 x 3)
        if results.right_hand_landmarks:
            keypoints.extend(np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).flatten())
        else:
            keypoints.extend(np.zeros(21 * 3))

        keypoints_all_frames.append(keypoints)
        frame_count += 1

    cap.release()
    holistic.close()
    return np.array(keypoints_all_frames)

def process_all_videos():
    """Xử lý tất cả video trong thư mục và lưu keypoints"""
    gesture_folders = [d for d in os.listdir(VIDEO_ROOT_DIR) 
                      if os.path.isdir(os.path.join(VIDEO_ROOT_DIR, d))]
    
    for gesture_folder in tqdm(gesture_folders, desc="Processing gestures"):
        gesture_dir = os.path.join(VIDEO_ROOT_DIR, gesture_folder)
        video_files = [f for f in os.listdir(gesture_dir) if f.endswith('.mp4')]
        all_keypoints = []
        
        for video_file in tqdm(video_files, desc=f"Videos in {gesture_folder}", leave=False):
            video_path = os.path.join(gesture_dir, video_file)
            try:
                keypoints = extract_keypoints_from_video(video_path)
                if len(keypoints) > 0:  # Chỉ thêm nếu có keypoints
                    all_keypoints.append(keypoints)
            except Exception as e:
                print(f"\nError processing {video_path}: {str(e)}")
                continue
        
        if all_keypoints:
            # Lưu mỗi video thành file riêng hoặc gộp thành 1 file
            # Ở đây tôi lưu mỗi video thành file riêng với tên gesture_folder_X.npy
            for i, kp in enumerate(all_keypoints):
                output_file = os.path.join(OUTPUT_DIR, f"{gesture_folder}_{i+1}.npy")
                np.save(output_file, kp)
                print(f"\u2714 Saved {output_file} with shape {kp.shape}")

if __name__ == '__main__':
    process_all_videos()
    print("Hoàn thành trích xuất keypoints cho tất cả video!")
