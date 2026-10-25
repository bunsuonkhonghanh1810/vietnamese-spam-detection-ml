# -*- coding: utf-8 -*-
"""
app.py
Giao diện ứng dụng Web: Hệ thống phân loại Tin nhắn & Email rác tiếng Việt.
Tập trung trọn vẹn vào phần Live Demo (Tái hiện Figure 8, 9, 10 trong bài báo IEEE ICSSIT 2023).
"""

import os
import sys
import pickle
import streamlit as st

# Thêm đường dẫn thư mục nguồn vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocess import clean_text

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Phân loại tin nhắn rác tiếng Việt",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Đường dẫn tài nguyên mô hình
base_dir = os.path.dirname(current_dir)
models_dir = os.path.join(base_dir, "models")
vec_path = os.path.join(models_dir, "vectorizer.pkl")
model_path = os.path.join(models_dir, "model.pkl")

@st.cache_resource
def load_artifacts():
    """Tải bộ trích xuất đặc trưng TF-IDF và mô hình phân loại."""
    try:
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        return None, None

vectorizer, model = load_artifacts()

# Danh mục mẫu thử nghiệm thực tế
SAMPLE_MESSAGES = {
    "-- Chọn tin nhắn mẫu để kiểm tra nhanh --": "",
    "[Spam] Giả mạo ngân hàng thông báo khóa tài khoản": "Canh bao tu Vietcombank: Tai khoan cua ban dang bi khoa do het han xac thuc. Vui long truy cap https://vietcombank-online.com de xac thuc ngay.",
    "[Ham] Mã xác thực OTP ngân hàng hợp lệ": "Ma OTP xac thuc GD la 066595, hieu luc trong 3 phut. Chi tiet GD: Chuyen khoan nhanh qua so TK, so tien 500.000 VND. Tuyet doi khong chia se ma nay.",
    "[Spam] Lừa đảo trúng thưởng xe máy / tiền mặt": "Chuc mung Quy khach da may man trung thuong 1 chiec xe may SH 150i tu chuong trinh tri an khach hang. Lien he hotline 0987654321 de nhan thuong ngay.",
    "[Ham] Thông báo cước viễn thông hàng tháng": "[TB] Cuoc thue bao thang nay cua Quy khach la 120.000d. Quy khach vui long thanh toan truoc ngay 20/10 de khong bi chan cuoc. Chi tiet lien he 198 (mien phi).",
    "[Spam] Lừa đảo tuyển dụng làm việc online hoa hồng cao": "Tuyen dung cong tac vien online lam viec tai nha, chi can like video va danh gia san pham, thu nhap 300k - 500k/ngay, nhan tien ngay trong ngay. Nhan vao link de tham gia."
}

# Tiêu đề giao diện
st.markdown("<h3>Hệ thống phân loại tin nhắn rác tiếng Việt</h3>", unsafe_allow_html=True)
st.markdown("---")

col_input, col_sample = st.columns([3, 2])

with col_sample:
    selected_sample = st.selectbox(
        "Mẫu thử nghiệm nhanh:",
        list(SAMPLE_MESSAGES.keys()),
        label_visibility="visible"
    )
    initial_text = SAMPLE_MESSAGES[selected_sample]

with col_input:
    user_input = st.text_area(
        "Nội dung tin nhắn (SMS / Email):",
        value=initial_text,
        height=140,
        placeholder="Dán hoặc nhập nội dung tin nhắn cần kiểm tra vào đây..."
    )

predict_btn = st.button("Phân loại tin nhắn (Predict)", type="primary")

if predict_btn:
    if not user_input.strip():
        st.warning("Vui lòng nhập nội dung tin nhắn trước khi thực hiện dự đoán.")
    elif vectorizer is None or model is None:
        st.error("Không tìm thấy tệp mô hình đã huấn luyện (vectorizer.pkl / model.pkl).")
    else:
        # 1. Pipeline tiền xử lý văn bản tiếng Việt
        cleaned = clean_text(user_input)
        
        # 2. Vector hóa TF-IDF
        vec_input = vectorizer.transform([cleaned])
        
        # 3. Dự đoán nhãn và xác suất
        prediction = model.predict(vec_input)[0]
        proba = model.predict_proba(vec_input)[0]
        spam_prob = proba[1] * 100
        ham_prob = proba[0] * 100

        st.markdown("---")
        st.markdown("##### Kết quả phân loại")

        col_res1, col_res2, col_res3 = st.columns([2, 1, 1])

        with col_res1:
            # Fig 9: Kết quả Spam
            if prediction == 1:
                st.error("KẾT QUẢ: TIN NHẮN RÁC / LỪA ĐẢO (SPAM)")
                st.caption("Cảnh báo: Nội dung mang dấu hiệu lừa đảo, giả mạo danh tính hoặc chứa liên kết độc hại.")
            # Fig 10: Kết quả Ham
            else:
                st.success("KẾT QUẢ: TIN NHẮN HỢP LỆ (HAM / NOT SPAM)")
                st.caption("Nội dung thuộc nhóm tin nhắn trao đổi thông thường hoặc thông báo dịch vụ chuẩn mực.")

        with col_res2:
            st.metric("Xác suất Spam", f"{spam_prob:.2f}%")

        with col_res3:
            st.metric("Xác suất Ham", f"{ham_prob:.2f}%")

        # Thanh tiến trình xác suất
        st.progress(int(spam_prob) if prediction == 1 else int(ham_prob))

        # Thông tin tiền xử lý kỹ thuật
        with st.expander("Chi tiết tiền xử lý văn bản (NLP)"):
            st.write("**Văn bản sau khi chuẩn hóa và loại bỏ từ dừng:**")
            st.code(cleaned if cleaned else "(Không có từ khóa sau khi lọc)")
            st.write(f"- Chiều dài ký tự gốc: `{len(user_input)}` | Số từ: `{len(user_input.split())}`")
