# -*- coding: utf-8 -*-
"""
app.py
Giao diện ứng dụng Web: Hệ thống phân loại Tin nhắn & Email rác tiếng Việt.
Thiết kế tối giản, chuyên nghiệp phục vụ báo cáo khoa học và đồ án môn học.
Bám sát kiến trúc bài báo IEEE ICSSIT 2023 (Figure 8, 9, 10).
"""

import os
import sys
import pickle
import pandas as pd
import streamlit as st

# Thêm đường dẫn thư mục nguồn vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocess import clean_text

# Cấu hình giao diện Streamlit chuẩn doanh nghiệp / học thuật
st.set_page_config(
    page_title="Hệ thống phân loại tin nhắn rác tiếng Việt",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Đường dẫn tài nguyên
base_dir = os.path.dirname(current_dir)
models_dir = os.path.join(base_dir, "models")
fig_dir = os.path.join(models_dir, "figures")
vec_path = os.path.join(models_dir, "vectorizer.pkl")
model_path = os.path.join(models_dir, "model.pkl")
comparison_csv = os.path.join(models_dir, "model_comparison.csv")

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

# Danh mục mẫu thử nghiệm thực tế (không dùng emoji rườm rà)
SAMPLE_MESSAGES = {
    "-- Chọn tin nhắn mẫu để kiểm tra nhanh --": "",
    "[Spam] Giả mạo ngân hàng thông báo khóa tài khoản": "Canh bao tu Vietcombank: Tai khoan cua ban dang bi khoa do het han xac thuc. Vui long truy cap https://vietcombank-online.com de xac thuc ngay.",
    "[Ham] Mã xác thực OTP ngân hàng hợp lệ": "Ma OTP xac thuc GD la 066595, hieu luc trong 3 phut. Chi tiet GD: Chuyen khoan nhanh qua so TK, so tien 500.000 VND. Tuyet doi khong chia se ma nay.",
    "[Spam] Lừa đảo trúng thưởng xe máy / tiền mặt": "Chuc mung Quy khach da may man trung thuong 1 chiec xe may SH 150i tu chuong trinh tri an khach hang. Lien he hotline 0987654321 de nhan thuong ngay.",
    "[Ham] Thông báo cước viễn thông hàng tháng": "[TB] Cuoc thue bao thang nay cua Quy khach la 120.000d. Quy khach vui long thanh toan truoc ngay 20/10 de khong bi chan cuoc. Chi tiet lien he 198 (mien phi).",
    "[Spam] Lừa đảo tuyển dụng làm việc online hoa hồng cao": "Tuyen dung cong tac vien online lam viec tai nha, chi can like video va danh gia san pham, thu nhap 300k - 500k/ngay, nhan tien ngay trong ngay. Nhan vao link de tham gia."
}

# Tiêu đề & Thông tin đề tài
st.markdown("<h2 style='margin-bottom: 0px;'>Hệ thống phân loại tin nhắn rác tiếng Việt</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #666; font-size: 14px;'>Đồ án Machine Learning | TS. Lương Thái Lê | Nghiên cứu thực nghiệm theo bài báo IEEE ICSSIT 2023</p>", unsafe_allow_html=True)
st.markdown("---")

# Phân chia các tab chức năng
tab1, tab2, tab3 = st.tabs([
    "Kiểm tra tin nhắn (Fig 8, 9, 10)",
    "Đối chuẩn 10 mô hình học máy (Fig 7)",
    "Phân tích khám phá dữ liệu (Fig 3 - 6)"
])

# =========================================================================
# TAB 1: KIỂM TRA TIN NHẮN (TÁI HIỆN FIG 8, 9, 10)
# =========================================================================
with tab1:
    st.markdown("##### Nhập nội dung tin nhắn cần phân loại")
    
    col_input, col_sample = st.columns([3, 2])
    
    with col_sample:
        selected_sample = st.selectbox(
            "Chọn mẫu tin nhắn thực nghiệm:",
            list(SAMPLE_MESSAGES.keys()),
            label_visibility="visible"
        )
        initial_text = SAMPLE_MESSAGES[selected_sample]

    with col_input:
        user_input = st.text_area(
            "Nội dung văn bản (SMS / Email):",
            value=initial_text,
            height=130,
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

            # Tiến trình trực quan
            st.progress(int(spam_prob) if prediction == 1 else int(ham_prob))

            # Thông số xử lý NLP
            with st.expander("Thông tin xử lý ngôn ngữ tự nhiên (NLP)"):
                st.write("**Văn bản sau khi chuẩn hóa và loại bỏ từ dừng:**")
                st.code(cleaned if cleaned else "(Không có từ khóa sau khi lọc)")
                st.write(f"- Độ dài ký tự gốc: `{len(user_input)}` | Số từ: `{len(user_input.split())}`")

# =========================================================================
# TAB 2: ĐỐI CHUẨN MÔ HÌNH (TÁI HIỆN FIG 7)
# =========================================================================
with tab2:
    st.markdown("##### Đối chuẩn hiệu năng 10 giải thuật học máy (Figure 7)")
    st.caption("Đánh giá độc lập trên tập kiểm thử 598 tin nhắn SMS tiếng Việt (tỉ lệ 80/20 phân tầng):")

    col_tbl, col_chart = st.columns([1, 1])

    with col_tbl:
        if os.path.exists(comparison_csv):
            df_comp = pd.read_csv(comparison_csv)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            st.markdown("""
            **Phân tích kỹ thuật (Bám sát Slide L2 & L4)**:
            * **Multinomial Naïve Bayes**: Đạt Accuracy 92.47% và Precision 84.85%. Ưu thế vượt trội về tốc độ suy luận nhanh (dưới 1 ms), tiêu thụ ít bộ nhớ RAM và kích thước file mô hình chỉ 96 KB.
            * **Extra Trees & Random Forest**: Đạt độ chính xác và Precision cao nhất (~97 - 98%) trên ma trận đặc trưng TF-IDF.
            * **K-Nearest Neighbors (KNN)**: Hiệu năng giảm sâu (Accuracy 71.24%, Precision 48.19%) do hiện tượng suy giảm khoảng cách trong không gian 3.000 chiều (Curse of Dimensionality).
            """)
        else:
            st.warning("Chưa tìm thấy tệp dữ liệu model_comparison.csv.")

    with col_chart:
        fig7_file = os.path.join(fig_dir, "fig7_model_comparison.png")
        if os.path.exists(fig7_file):
            st.image(fig7_file, caption="Fig 7. Model comparison based on performance (Accuracy vs Precision)", use_column_width=True)
        else:
            st.warning("Chưa tìm thấy tệp ảnh fig7_model_comparison.png.")

# =========================================================================
# TAB 3: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (TÁI HIỆN FIG 3 - 6)
# =========================================================================
with tab3:
    st.markdown("##### Khám phá dữ liệu (Exploratory Data Analysis - EDA)")
    st.caption("Các biểu đồ phân tích trực quan bộ dữ liệu 2.991 tin nhắn thực tế tại Việt Nam:")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        f3 = os.path.join(fig_dir, "fig3_target_distribution_pie.png")
        if os.path.exists(f3):
            st.image(f3, caption="Fig 3. Dataset classifying spam and ham", use_column_width=True)

    with row1_c2:
        f4 = os.path.join(fig_dir, "fig4_correlation_heatmap.png")
        if os.path.exists(f4):
            st.image(f4, caption="Fig 4. Correlation Heatmap of Structural Features", use_column_width=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        f5 = os.path.join(fig_dir, "fig5_spam_wordcloud.png")
        if os.path.exists(f5):
            st.image(f5, caption="Fig 5. Word Cloud of Vietnamese Spam Messages", use_column_width=True)

    with row2_c2:
        f6 = os.path.join(fig_dir, "fig6_ham_wordcloud.png")
        if os.path.exists(f6):
            st.image(f6, caption="Fig 6. Word Cloud of Vietnamese Ham Messages", use_column_width=True)
