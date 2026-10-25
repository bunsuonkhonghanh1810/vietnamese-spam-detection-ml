# -*- coding: utf-8 -*-
"""
preprocess.py
Module tiền xử lý văn bản tiếng Việt cho bài toán phân loại Tin nhắn & Email rác.
Tái hiện quy trình Preprocessing của bài báo IEEE ICSSIT 2023:
1. Chuẩn hóa bảng mã Unicode về dạng chuẩn NFC
2. Chuyển đổi chữ thường (Lowercasing)
3. Quy chuẩn các thực thể: URLs, Email, Số điện thoại, Tiền tệ, Ngày tháng
4. Loại bỏ dấu câu và ký tự đặc biệt
5. Loại bỏ từ dừng tiếng Việt (Stopwords)
6. Tách từ và chuẩn hóa khoảng trắng
"""

import os
import re
import sys
import unicodedata
import pandas as pd

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi bảng mã cp1252
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Danh sách từ dừng (Stopwords) tiếng Việt cơ bản không mang giá trị phân biệt spam/ham
VIETNAMESE_STOPWORDS = {
    "là", "và", "của", "có", "các", "được", "cho", "trong", "này", "với",
    "đã", "thì", "sẽ", "những", "để", "ở", "từ", "khi", "như", "người",
    "vào", "ra", "lại", "đến", "bởi", "về", "đó", "cũng", "hay", "hoặc",
    "nếu", "bị", "vì", "do", "nhưng", "mà", "lên", "xuống", "rất", "thế",
    "nào", "gì", "ai", "đâu", "đây", "tại", "qua", "trên", "dưới", "thêm"
}

def normalize_unicode(text: str) -> str:
    """Chuẩn hóa bảng mã tiếng Việt về Unicode dựng sẵn (NFC)."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Quy trình làm sạch văn bản tiếng Việt:
    - Chuẩn hóa Unicode
    - Chuyển chữ thường
    - Thay thế link URL, email, số điện thoại, tiền tệ thành token định danh
    - Loại bỏ dấu câu, ký tự đặc biệt
    - Xóa từ dừng và từ quá ngắn (<= 1 ký tự)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Chuẩn hóa Unicode
    text = normalize_unicode(text)
    
    # 2. Chuyển về chữ thường
    text = text.lower()
    
    # 3. Chuẩn hóa URL
    text = re.sub(r'https?://\S+|www\.\S+', ' url_link ', text)
    
    # 4. Chuẩn hóa Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_addr ', text)
    
    # 5. Chuẩn hóa số điện thoại (đầu số Việt Nam: 03, 05, 07, 08, 09 hoặc +84)
    text = re.sub(r'(\+?84|0)(3[2-9]|5[6|8|9]|7[0|6-9]|8[1-9]|9[0-9])[0-9]{7}', ' phone_num ', text)
    
    # 6. Chuẩn hóa các token có sẵn trong bộ dữ liệu
    text = text.replace('[money]', ' tien_vnd ')
    text = text.replace('[number]', ' so_luong ')
    text = text.replace('[date]', ' ngay_thang ')
    text = text.replace('[time]', ' thoi_gian ')
    
    # 7. Loại bỏ dấu câu và ký tự đặc biệt (giữ lại chữ cái tiếng Việt và số)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 8. Tách từ và loại bỏ khoảng trắng thừa
    words = text.split()
    
    # 9. Lọc từ dừng và từ đơn vô nghĩa
    if remove_stopwords:
        words = [w for w in words if w not in VIETNAMESE_STOPWORDS and len(w) > 1]
    else:
        words = [w for w in words if len(w) > 1]
        
    return " ".join(words)

def process_and_save_dataset(input_csv: str, output_csv: str):
    """Đọc dữ liệu thô, áp dụng tiền xử lý và lưu kết quả vào thư mục processed."""
    print(f"-> Đang đọc dữ liệu thô từ: {input_csv}")
    df = pd.read_csv(input_csv)
    initial_rows = len(df)
    
    print(f"-> Tổng số tin nhắn ban đầu: {initial_rows}")
    print("-> Đang thực thi pipeline tiền xử lý văn bản tiếng Việt...")
    
    # Tiền xử lý văn bản
    df["cleaned_message"] = df["message"].apply(clean_text)
    
    # Thống kê độ dài phục vụ phân tích
    df["raw_char_len"] = df["message"].astype(str).apply(len)
    df["raw_word_count"] = df["message"].astype(str).apply(lambda x: len(x.split()))
    df["cleaned_word_count"] = df["cleaned_message"].apply(lambda x: len(x.split()))
    
    # Loại bỏ các dòng rỗng sau khi làm sạch nếu có
    df = df[df["cleaned_message"].str.strip().str.len() > 0].copy()
    final_rows = len(df)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    
    print(f"-> Tiền xử lý hoàn tất! Đã lưu: {output_csv}")
    print(f"-> Số tin nhắn sau làm sạch: {final_rows} (loại bỏ {initial_rows - final_rows} dòng rỗng)")
    print(f"   + Số tin nhắn hợp lệ (Ham - 0): {sum(df['label'] == 0)}")
    print(f"   + Số tin nhắn rác (Spam - 1):   {sum(df['label'] == 1)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw", "vietnamese_sms_spam.csv")
    processed_path = os.path.join(base_dir, "data", "processed", "vietnamese_sms_spam_cleaned.csv")
    
    process_and_save_dataset(raw_path, processed_path)
