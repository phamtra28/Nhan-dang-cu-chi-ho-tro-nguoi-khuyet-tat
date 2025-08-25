import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

KEYPOINT_DIR = 'gesture_data/keypoints'
OUTPUT_DIR = 'gesture_data'
SEQUENCE_LENGTH = 30

def prepare_dataset():
    X = []
    y = []
    action_map = {}  # Ánh xạ tên cử chỉ -> số
    all_actions = set()  # Tập hợp tất cả các cử chỉ
    
    # Bước 1: Thu thập tất cả các cử chỉ duy nhất
    for filename in os.listdir(KEYPOINT_DIR):
        if filename.endswith('.npy'):
            # Tách tên cử chỉ từ filename (format: GESTURE_X.npy)
            action_name = '_'.join(filename.split('_')[:-1])
            all_actions.add(action_name)
    
    if not all_actions:
        print("❌ Không tìm thấy file .npy nào trong thư mục keypoints")
        return
    
    all_actions = sorted(list(all_actions))  # Chuyển thành list và sắp xếp
    action_map = {action: idx for idx, action in enumerate(all_actions)}
    
    print(f"📌 Tìm thấy {len(all_actions)} cử chỉ: {all_actions}")

    # Bước 2: Xử lý từng file
    valid_files = 0
    for filename in tqdm(os.listdir(KEYPOINT_DIR), desc="Processing files"):
        if not filename.endswith('.npy'):
            continue
        
        # Lấy tên cử chỉ từ tên file
        action_name = '_'.join(filename.split('_')[:-1])
        if action_name not in action_map:
            continue
        
        file_path = os.path.join(KEYPOINT_DIR, filename)
        keypoints = np.load(file_path)
        
        # Xử lý sequence length
        if keypoints.shape[0] < SEQUENCE_LENGTH:
            # Padding với zeros nếu thiếu
            pad_shape = (SEQUENCE_LENGTH - keypoints.shape[0],) + keypoints.shape[1:]
            padding = np.zeros(pad_shape)
            keypoints = np.concatenate((keypoints, padding), axis=0)
        elif keypoints.shape[0] > SEQUENCE_LENGTH:
            # Lấy mẫu đều nếu dư
            indices = np.linspace(0, keypoints.shape[0]-1, num=SEQUENCE_LENGTH, dtype=int)
            keypoints = keypoints[indices]
        
        # Kiểm tra shape cuối cùng
        if keypoints.shape != (SEQUENCE_LENGTH, 1662):
            print(f"\n⚠️ Bỏ qua {filename} do shape không hợp lệ: {keypoints.shape}")
            continue
        
        X.append(keypoints)
        y.append(action_map[action_name])
        valid_files += 1
    
    # Bước 3: Lưu dữ liệu
    if not X:
        print("❌ Không có dữ liệu hợp lệ nào được tạo!")
        return
    
    X = np.array(X)
    y = to_categorical(y)
    
    print("\n📊 Thống kê dataset:")
    print(f"- Số lượng mẫu: {len(X)}")
    print(f"- Shape X: {X.shape}")
    print(f"- Shape y: {y.shape}")
    print(f"- Số cử chỉ: {len(all_actions)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    np.save(os.path.join(OUTPUT_DIR, 'action_map.npy'), np.array(all_actions))
    
    print("\n✅ Đã lưu dataset thành công!")
    print(f"- X.npy: {X.shape}")
    print(f"- y.npy: {y.shape}")
    print(f"- action_map.npy: {len(all_actions)} nhãn")

if __name__ == '__main__':
    prepare_dataset()
