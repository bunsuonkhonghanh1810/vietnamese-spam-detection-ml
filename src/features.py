# -*- coding: utf-8 -*-
"""
features.py
Module trích xuất đặc trưng vector hóa bằng phương pháp TF-IDF (Term Frequency - Inverse Document Frequency).
Tái hiện kỹ thuật Vector hóa văn bản trong bài báo IEEE ICSSIT 2023:
- Phân chia tập dữ liệu: 80% Train - 20% Test phân tầng (Stratified)
- N-gram Range: (1, 2) - Kết hợp cả từ đơn (Unigram) và cụm 2 từ (Bigram tiếng Việt)
- Sublinear TF: Áp dụng tỉ lệ logarithmic để cân bằng trọng số từ xuất hiện nhiều lần
- Cố định random_state=42 đảm bảo tính tái lập theo Slide L1 & Slide L2 của TS. Lương Thái Lê
"""

import os
import sys
import pandas as pd
from typing import Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Cấu hình encoding console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def load_processed_data(csv_path: str = None) -> Tuple[pd.Series, pd.Series]:
    """Tải tập dữ liệu đã làm sạch từ thư mục processed."""
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "processed", "vietnamese_sms_spam_cleaned.csv")
        
    df = pd.read_csv(csv_path)
    # Đảm bảo không có giá trị NaN trong cột văn bản
    df = df.dropna(subset=["cleaned_message", "label"]).copy()
    
    X = df["cleaned_message"].astype(str)
    y = df["label"].astype(int)
    return X, y

def split_data(X: pd.Series, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Phân chia Train/Test phân tầng (Stratified) theo đúng tỉ lệ nhãn.
    Tránh hiện tượng mất cân bằng nhãn ở tập kiểm thử.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def get_tfidf_vectorizer(
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> TfidfVectorizer:
    """
    Khởi tạo bộ vectorizer TF-IDF tối ưu cho tiếng Việt:
    - ngram_range=(1, 2): Bắt được các cụm từ đắt giá như 'mã otp', 'tài khoản', 'chuyển khoản'
    - min_df=2: Loại bỏ từ hiếm/nhiễu chỉ xuất hiện 1 lần
    - sublinear_tf=True: Thay thế tf bằng 1 + log(tf), giảm ảnh hưởng của việc lặp từ
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True
    )

def extract_tfidf_features(
    X_train: pd.Series,
    X_test: pd.Series,
    vectorizer: TfidfVectorizer = None
) -> Tuple[Any, Any, TfidfVectorizer]:
    """
    Thực hiện trích xuất đặc trưng:
    - fit_transform() CHỈ trên tập Train
    - transform() trên tập Test
    => Đảm bảo tuyệt đối KHÔNG xảy ra rò rỉ dữ liệu (Data Leakage)
    """
    if vectorizer is None:
        vectorizer = get_tfidf_vectorizer()
        
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    return X_train_vec, X_test_vec, vectorizer

def test_feature_pipeline():
    """Hàm chạy kiểm thử pipeline trích xuất đặc trưng."""
    print("=" * 70)
    print(" BẮT ĐẦU KIỂM THỬ PIPELINE TRÍCH XUẤT ĐẶC TRƯNG TF-IDF - TASK 4")
    print("=" * 70)
    
    # 1. Nạp dữ liệu
    X, y = load_processed_data()
    print(f"-> Tổng số mẫu: {len(X)}")
    print(f"-> Tỷ lệ nhãn: Ham (0) = {sum(y == 0)}, Spam (1) = {sum(y == 1)}")
    
    # 2. Phân chia dữ liệu
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"-> Tập Train: {len(X_train)} mẫu (Ham: {sum(y_train == 0)}, Spam: {sum(y_train == 1)})")
    print(f"-> Tập Test:  {len(X_test)} mẫu (Ham: {sum(y_test == 0)}, Spam: {sum(y_test == 1)})")
    
    # 3. Vector hóa TF-IDF
    X_train_vec, X_test_vec, tfidf = extract_tfidf_features(X_train, X_test)
    print(f"-> Ma trận đặc trưng Train: {X_train_vec.shape[0]} dòng x {X_train_vec.shape[1]} đặc trưng TF-IDF")
    print(f"-> Ma trận đặc trưng Test:  {X_test_vec.shape[0]} dòng x {X_test_vec.shape[1]} đặc trưng TF-IDF")
    print(f"-> Mật độ ma trận thưa (Sparsity): {(1.0 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1])) * 100:.2f}%")
    
    # In một số đặc trưng tiêu biểu
    feature_names = tfidf.get_feature_names_out()
    print(f"-> 15 đặc trưng tiêu biểu trong từ điển: {list(feature_names[50:65])}")
    
    print("\n" + "=" * 70)
    print(" HOÀN TẤT TASK 4! Pipeline trích xuất đặc trưng TF-IDF hoạt động hoàn hảo.")
    print("=" * 70)

if __name__ == "__main__":
    test_feature_pipeline()
