import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

# Load dữ liệu
X = np.load('gesture_data/X.npy')
y = np.load('gesture_data/y.npy')
action_list = np.load('gesture_data/action_map.npy', allow_pickle=True)

print("\n📊 Thông tin dataset:")
print(f"- Số mẫu huấn luyện: {X.shape[0]}")
print(f"- Số cử chỉ: {len(action_list)}")
print(f"- Các cử chỉ: {', '.join(action_list)}")

# Chia dữ liệu thành train-val-test (60-20-20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\n🔢 Phân chia dữ liệu:")
print(f"- Train: {X_train.shape[0]} mẫu")
print(f"- Validation: {X_val.shape[0]} mẫu")
print(f"- Test: {X_test.shape[0]} mẫu")

# Tạo mô hình LSTM cải tiến
model = Sequential([
    Input(shape=(30, 1662)),
    BatchNormalization(),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dense(len(action_list), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

print("\n🧠 Thông tin mô hình:")
model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
]

# Huấn luyện mô hình
print("\n🏋️ Bắt đầu huấn luyện...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Đánh giá trên tập test
print("\n🧪 Đánh giá trên tập test:")
test_loss, test_acc, test_top_k = model.evaluate(X_test, y_test, verbose=0)
print(f"- Test Accuracy: {test_acc:.4f}")
print(f"- Test Top-3 Accuracy: {test_top_k:.4f}")
print(f"- Test Loss: {test_loss:.4f}")

# Lưu mô hình cuối cùng
model.save("final_model.keras")
print("\n✅ Đã lưu mô hình thành công:")
print("- best_model.keras (mô hình tốt nhất trên validation)")
print("- final_model.keras (mô hình sau khi huấn luyện xong)")
