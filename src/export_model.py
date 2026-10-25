# -*- coding: utf-8 -*-
"""
export_model.py
Module đóng gói mô hình học máy tối ưu (Model Serialization) bằng thư viện pickle.
Tái hiện quy trình trong bài báo IEEE ICSSIT 2023:
1. Huấn luyện bộ trích xuất đặc trưng TF-IDF và mô hình Multinomial Naïve Bayes
2. Đóng gói ra 2 tệp nhị phân:
   - models/vectorizer.pkl: Chứa từ điển từ vựng và trọng số TF-IDF đã học
   - models/model.pkl: Chứa mô hình Multinomial Naïve Bayes đã tối ưu hóa
3. Kiểm thử tải lại (Sanity Check / Smoke Test) với các mẫu tin nhắn mới trong thực tế
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# Cấu hình encoding console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import các hàm từ src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocess import clean_text
from features import load_processed_data, split_data, get_tfidf_vectorizer

def export_and_test():
    base_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    vec_path = os.path.join(models_dir, "vectorizer.pkl")
    model_path = os.path.join(models_dir, "model.pkl")

    print("=" * 75)
    print(" BẮT ĐẦU ĐÓNG GÓI MÔ HÌNH HỌC MÁY (TASK 6 - MODEL SERIALIZATION)")
    print("=" * 75)

    # 1. Nạp dữ liệu và phân chia
    print("[1/3] Đang nạp dữ liệu và chuẩn bị tập huấn luyện...")
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 2. Vector hóa và huấn luyện Naïve Bayes
    print("[2/3] Huấn luyện TF-IDF Vectorizer và Multinomial Naïve Bayes...")
    tfidf = get_tfidf_vectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)

    # Tinh chỉnh siêu tham số alpha=0.1 theo Slide L4 (Laplace Smoothing)
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_vec, y_train)

    # Đóng gói ra file .pkl
    print(" -> Đang đóng gói ra models/vectorizer.pkl...")
    with open(vec_path, "wb") as f:
        pickle.dump(tfidf, f)

    print(" -> Đang đóng gói ra models/model.pkl...")
    with open(model_path, "wb") as f:
        pickle.dump(nb_model, f)

    print(f" -> Hoàn tất! Dung lượng vectorizer.pkl: {os.path.getsize(vec_path) / 1024:.1f} KB")
    print(f" -> Hoàn tất! Dung lượng model.pkl:      {os.path.getsize(model_path) / 1024:.1f} KB")

    # 3. Kiểm thử tải lại mô hình (Sanity Check) với dữ liệu mới
    print("\n[3/3] Kiểm thử tải lại mô hình từ file .pkl với các tin nhắn mẫu thực tế...")
    with open(vec_path, "rb") as f:
        loaded_vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    test_messages = [
        # Mẫu 1: Tin nhắn lừa đảo ngân hàng
        "Canh bao: Tai khoan Vietcombank cua ban dang bi khoa. Vui long dang nhap tai https://vcb-online.com de xac thuc ngay",
        # Mẫu 2: Tin nhắn mã OTP ngân hàng hợp lệ
        "Ma OTP xac thuc giao dich chuyen khoan cua Quy khach la 582910. Ma co hieu luc trong 3 phut. Tuyet doi khong chia se cho bat ky ai.",
        # Mẫu 3: Tin nhắn lừa đảo trúng thưởng / cờ bạc
        "Chuc mung ban da trung thuong 100 trieu dong tu chuong trinh tri an khach hang. Nhan vao day de rut tien ve the",
        # Mẫu 4: Tin nhắn cước nhà mạng hợp lệ
        "[TB] Cuoc thue bao thang nay cua Quy khach la 120.000d. QK vui long thanh toan truoc ngay 20 de khong bi chan 1 chieu. Chi tiet LH 198."
    ]

    print("-" * 75)
    for idx, msg in enumerate(test_messages, 1):
        # Pipeline: Clean -> Transform -> Predict
        cleaned = clean_text(msg)
        vec_input = loaded_vectorizer.transform([cleaned])
        pred = loaded_model.predict(vec_input)[0]
        prob = loaded_model.predict_proba(vec_input)[0]
        
        status = "🚨 SPAM (RÁC / LỪA ĐẢO)" if pred == 1 else "✅ HAM (HỢP LỆ)"
        confidence = prob[pred] * 100

        print(f"Mẫu {idx}: \"{msg[:65]}...\"")
        print(f"  => Dự đoán: {status} | Độ tin cậy: {confidence:.2f}%\n")

    print("=" * 75)
    print(" HOÀN TẤT TASK 6! Hai tệp nhị phân đã sẵn sàng để tích hợp vào ứng dụng Web.")
    print("=" * 75)

if __name__ == "__main__":
    export_and_test()
