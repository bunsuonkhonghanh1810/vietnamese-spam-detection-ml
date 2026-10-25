# -*- coding: utf-8 -*-
"""
app.py
Ứng dụng Web tương tác Streamlit: Hệ thống phân loại Tin nhắn & Email rác tiếng Việt.
Tái hiện 100% giao diện và trải nghiệm trong bài báo IEEE ICSSIT 2023:
- Fig 8: Giao diện nhập tin nhắn và nút bấm kiểm tra (Classifier Window)
- Fig 9: Kết quả hiển thị cảnh báo tin nhắn Spam (The result showing spam SMS)
- Fig 10: Kết quả hiển thị thông báo tin nhắn Ham (The result showing ham SMS)
"""

import os
import sys
import pickle
import pandas as pd
import streamlit as st

# Thêm đường dẫn src vào hệ thống
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from preprocess import clean_text

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống lọc Spam SMS & Email Tiếng Việt",
    page_icon="🛡️",
    layout="wide"
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
    """Tải bộ trích xuất đặc trưng TF-IDF và mô hình Naïve Bayes từ file pickle."""
    try:
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None, None

vectorizer, model = load_artifacts()

# Danh sách mẫu tin nhắn thử nghiệm nhanh
SAMPLE_MESSAGES = {
    "-- Chọn một tin nhắn mẫu thực tế để thử nhanh --": "",
    "🚨 [Spam] Giả mạo Vietcombank khóa tài khoản": "Canh bao tu Vietcombank: Tai khoan cua ban dang bi khoa do het han xac thuc. Vui long truy cap https://vietcombank-online.com de xac thuc ngay.",
    "✅ [Ham] Mã OTP ngân hàng hợp lệ": "Ma OTP xac thuc GD la 066595, hieu luc trong 3 phut. Chi tiet GD: Chuyen khoan nhanh qua so TK, so tien 500.000 VND. Tuyet doi khong chia se ma nay.",
    "🚨 [Spam] Lừa đảo trúng thưởng xe máy / tiền mặt": "Chuc mung Quy khach da may man trung thuong 1 chiec xe may SH 150i tu chuong trinh tri an khach hang. Lien he hotline 0987654321 de nhan thuong ngay.",
    "✅ [Ham] Thông báo cước nhà mạng Viettel": "[TB] Cuoc thue bao thang nay cua Quy khach la 120.000d. Quy khach vui long thanh toan truoc ngay 20/10 de khong bi chan cuoc. Chi tiet lien he 198 (mien phi).",
    "🚨 [Spam] Lừa đảo việc làm online hoa hồng cao": "Tuyen dung cong tac vien online lam viec tai nha, chi can like video va danh gia san pham, thu nhap 300k - 500k/ngay, nhan tien ngay trong ngay. Nhan vao link de tham gia."
}

# Tiêu đề ứng dụng
st.title("🛡️ Hệ thống Lọc & Phân loại Tin nhắn, Email Rác Tiếng Việt")
st.caption("Dự án Học máy cơ bản (Machine Learning) - Giảng viên: TS. Lương Thái Lê | Tái hiện bài báo IEEE ICSSIT 2023")

# Thiết lập các Tab chức năng
tab1, tab2, tab3 = st.tabs([
    "🔍 Kiểm tra tin nhắn (Fig 8, 9, 10)",
    "📊 Đối chuẩn 10 mô hình ML (Fig 7)",
    "📈 Phân tích khám phá dữ liệu EDA (Fig 3-6)"
])

# =========================================================================
# TAB 1: KIỂM TRA TIN NHẮN THỰC TẾ (TÁI HIỆN FIG 8, 9, 10)
# =========================================================================
with tab1:
    st.subheader("Kiểm tra phân loại Spam / Ham theo thời gian thực")
    st.write("Nhập hoặc dán nội dung tin nhắn SMS hoặc Email vào khung bên dưới để hệ thống nhận diện tự động:")

    col_input, col_sample = st.columns([3, 2])
    
    with col_sample:
        selected_sample = st.selectbox("💡 Hoặc chọn nhanh mẫu tin nhắn thử nghiệm:", list(SAMPLE_MESSAGES.keys()))
        initial_text = SAMPLE_MESSAGES[selected_sample]

    with col_input:
        user_input = st.text_area(
            "Enter the message (Nhập nội dung tin nhắn):",
            value=initial_text,
            height=140,
            placeholder="Ví dụ: Tài khoản ngân hàng của bạn đang gặp sự cố, vui lòng đăng nhập..."
        )

    predict_btn = st.button("🔮 Predict (Kiểm tra ngay)", type="primary", use_container_width=True)

    if predict_btn:
        if not user_input.strip():
            st.warning("⚠️ Vui lòng nhập nội dung tin nhắn trước khi bấm Predict!")
        elif vectorizer is None or model is None:
            st.error("❌ Không tìm thấy tệp mô hình đã huấn luyện! Hãy chạy Task 6 để tạo mô hình.")
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

            st.write("---")
            # FIGURE 9: Kết quả Spam (Spam SMS Result)
            if prediction == 1:
                st.error(f"""
                ### 🚨 KẾT QUẢ: ĐÂY LÀ TIN NHẮN RÁC / LỪA ĐẢO (SPAM)
                * **Độ nguy hiểm**: **{spam_prob:.2f}%**
                * **Đánh giá**: Tin nhắn có dấu hiệu lừa đảo, giả mạo thương hiệu hoặc bẫy liên kết độc hại. Khuyến cáo **KHÔNG** nhấn vào đường link hoặc cung cấp thông tin cá nhân.
                """)
                st.progress(int(spam_prob))
            # FIGURE 10: Kết quả Ham (Ham SMS Result)
            else:
                st.success(f"""
                ### ✅ KẾT QUẢ: TIN NHẮN HỢP LỆ (NOT SPAM / HAM)
                * **Độ tin cậy an toàn**: **{ham_prob:.2f}%**
                * **Đánh giá**: Nội dung tin nhắn không có các đặc trưng nhận diện spam, thuộc nhóm tin nhắn giao dịch, thông báo hoặc trao đổi bình thường.
                """)
                st.progress(int(ham_prob))

            # Hiển thị thông tin phân tích kỹ thuật
            with st.expander("🔬 Xem chi tiết phân tích xử lý ngôn ngữ tự nhiên (NLP)"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Văn bản sau khi tiền xử lý (Cleaned Tokens):**")
                    st.code(cleaned if cleaned else "(Không có từ khóa hợp lệ)")
                with c2:
                    st.write("**Xác suất phân phối của mô hình:**")
                    st.write(f"- Xác suất là Ham (An toàn): `{ham_prob:.2f}%`")
                    st.write(f"- Xác suất là Spam (Nguy hiểm): `{spam_prob:.2f}%`")

# =========================================================================
# TAB 2: ĐỐI CHUẨN 10 MÔ HÌNH HỌC MÁY (TÁI HIỆN FIG 7)
# =========================================================================
with tab2:
    st.subheader("Bảng đối chuẩn hiệu năng 10 thuật toán Machine Learning (Figure 7)")
    st.write("Đánh giá toàn diện trên tập kiểm thử độc lập (598 mẫu tiếng Việt) theo đúng phương pháp luận bài báo IEEE:")

    col_tbl, col_chart = st.columns([1, 1])

    with col_tbl:
        if os.path.exists(comparison_csv):
            df_comp = pd.read_csv(comparison_csv)
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
            st.info("""
            💡 **Nhận xét chuyên môn (Bám sát Slide L2 & L4)**:
            - **Multinomial Naïve Bayes** được chọn làm mô hình triển khai chính nhờ tốc độ suy luận mili-giây, dung lượng siêu nhẹ (96 KB) và không gian bộ nhớ thấp.
            - Các mô hình Ensemble (Extra Trees, Random Forest, SVC) đạt chỉ số Accuracy và Precision rất cao trên dữ liệu bảng vector TF-IDF.
            - **KNN** đạt hiệu năng thấp do hiện tượng lời nguyền số chiều (Curse of Dimensionality) trên không gian 3.000 đặc trưng thưa thớt.
            """)
        else:
            st.warning("Chưa có tệp model_comparison.csv.")

    with col_chart:
        fig7_file = os.path.join(fig_dir, "fig7_model_comparison.png")
        if os.path.exists(fig7_file):
            st.image(fig7_file, caption="Fig 7. Model comparison based on performance (Accuracy vs Precision)", use_container_width=True)
        else:
            st.warning("Chưa có ảnh Figure 7.")

# =========================================================================
# TAB 3: KHÁM PHÁ DỮ LIỆU EDA (TÁI HIỆN FIG 3, 4, 5, 6)
# =========================================================================
with tab3:
    st.subheader("Khám phá dữ liệu (Exploratory Data Analysis - EDA)")
    st.write("Các biểu đồ trực quan hóa bộ dữ liệu 2.991 tin nhắn SMS tiếng Việt thực tế:")

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        f3 = os.path.join(fig_dir, "fig3_target_distribution_pie.png")
        if os.path.exists(f3):
            st.image(f3, caption="Fig 3. Dataset classifying spam and ham", use_container_width=True)

    with row1_c2:
        f4 = os.path.join(fig_dir, "fig4_correlation_heatmap.png")
        if os.path.exists(f4):
            st.image(f4, caption="Fig 4. Correlation Heatmap of Structural Features", use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        f5 = os.path.join(fig_dir, "fig5_spam_wordcloud.png")
        if os.path.exists(f5):
            st.image(f5, caption="Fig 5. Word Cloud of Vietnamese Spam Messages", use_container_width=True)

    with row2_c2:
        f6 = os.path.join(fig_dir, "fig6_ham_wordcloud.png")
        if os.path.exists(f6):
            st.image(f6, caption="Fig 6. Word Cloud of Vietnamese Ham Messages", use_container_width=True)
