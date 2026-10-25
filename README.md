# Hệ thống Phân loại Tin nhắn Rác Tiếng Việt (Vietnamese Spam Detection)

Dự án nghiên cứu và triển khai hệ thống học máy tự động phát hiện và phân loại tin nhắn rác, tin nhắn lừa đảo viễn thông (SMS Phishing / Spam) và thư rác tiếng Việt.

Dự án tái hiện phương pháp luận từ bài báo khoa học:  
> **"Spam SMS (or) Email Detection and Classification using Machine Learning"**  
> *5th International Conference on Smart Systems and Inventive Technology (IEEE ICSSIT 2023)*  
> DOI: [10.1109/ICSSIT55814.2023.10060980](https://doi.org/10.1109/ICSSIT55814.2023.10060980)

---

## 1. Cấu trúc thư mục dự án

```text
ml/
├── data/
│   ├── raw/
│   │   └── vietnamese_sms_spam.csv          # Dữ liệu gốc 2.991 tin nhắn thực tế
│   ├── processed/
│   │   └── vietnamese_sms_spam_cleaned.csv  # Dữ liệu sau khi làm sạch NLP
│   └── README.md                            # Thuyết minh chi tiết về dữ liệu
├── docs/                                    # Hồ sơ đề tài, phân công & kế hoạch
│   ├── RULES.md                             # Quy tắc phát triển và chuẩn Git commit
│   ├── TASKS.md                             # Tiến độ 8 task độc lập
│   ├── Chi_Tiet_Nhiem_Vu_Thanh_Vien_3.docx  # Báo cáo chuyên sâu Thành viên 3
│   └── ...                                  # Các tài liệu phân công đồ án khác
├── models/
│   ├── figures/                             # Toàn bộ biểu đồ bài báo (Fig 3 - Fig 7)
│   │   ├── fig3_target_distribution_pie.png # Fig 3: Biểu đồ tròn tỷ lệ nhãn
│   │   ├── fig4_correlation_heatmap.png     # Fig 4: Heatmap tương quan độ dài
│   │   ├── fig5_spam_wordcloud.png          # Fig 5: Đám mây từ khóa Spam
│   │   ├── fig6_ham_wordcloud.png           # Fig 6: Đám mây từ khóa Ham
│   │   └── fig7_model_comparison.png        # Fig 7: Biểu đồ đối chuẩn 10 mô hình
│   ├── model_comparison.csv                 # Bảng số liệu chi tiết 10 thuật toán
│   ├── vectorizer.pkl                       # Trọng số TF-IDF đã huấn luyện
│   └── model.pkl                            # Mô hình phân loại đã đóng gói
├── paper/
│   └── 3-Spam_SMS_or_Email_Detection...pdf   # Bài báo khoa học gốc IEEE 2023
├── slide/                                   # Hệ thống slide bài giảng môn học
│   ├── L1_Overview.pdf
│   ├── L2_Problems in ML & Performance Evaluation.pdf
│   ├── L3_Decision Trees.pdf
│   ├── L4_Naive Bayes.pdf
│   ├── L5_SVMs.pdf
│   └── L6_Neural Network.pdf
├── src/
│   ├── preprocess.py                        # Pipeline tiền xử lý văn bản tiếng Việt
│   ├── eda.py                               # Phân tích khám phá & vẽ Fig 3 - 6
│   ├── features.py                          # Vector hóa đặc trưng TF-IDF
│   ├── train.py                             # Huấn luyện 10 mô hình & vẽ Fig 7
│   ├── export_model.py                      # Đóng gói vectorizer.pkl & model.pkl
│   └── app.py                               # Ứng dụng Web tương tác Streamlit
├── requirements.txt                         # Danh sách thư viện phụ thuộc
└── README.md                                # Tài liệu hướng dẫn tổng quan
```

---

## 2. Bộ dữ liệu thực nghiệm

* **Nguồn**: Bộ dữ liệu tin nhắn rác & lừa đảo tiếng Việt thực tế (*Quality-Assured Vietnamese SMS Phishing Dataset*).
* **Quy mô**: **2.991 tin nhắn** thu thập tại Việt Nam, đã ẩn danh thông tin nhạy cảm (PII Anonymized).
* **Phân bố nhãn**:
  * **Ham (`0` - Hợp lệ)**: 2.193 tin nhắn (tỷ lệ **73.32%**) - Mã OTP ngân hàng, biến động số dư, thông báo cước mạng, tin nhắn trao đổi.
  * **Spam (`1` - Rác / Lừa đảo)**: 798 tin nhắn (tỷ lệ **26.68%**) - Giả mạo ngân hàng (Vietcombank, ACB, BIDV), bẫy trúng thưởng, cờ bạc, việc làm online, link phishing.

---

## 3. Hướng dẫn cài đặt & Khởi chạy

### Cài đặt môi trường
Yêu cầu Python 3.9+ trở lên. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

### Khởi chạy ứng dụng Web phân loại tin nhắn
```bash
streamlit run src/app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 4. Quy trình kỹ thuật (Machine Learning Pipeline)

### Bước 1: Tiền xử lý văn bản tiếng Việt (`src/preprocess.py`)
1. **Chuẩn hóa Unicode**: Quy về chuẩn dựng sẵn (NFC).
2. **Chuyển chữ thường**: Lowercasing toàn bộ ký tự.
3. **Chuẩn hóa thực thể**: 
   * URL liên kết $\rightarrow$ token `url_link`
   * Số điện thoại $\rightarrow$ token `phone_num`
   * Email $\rightarrow$ token `email_addr`
   * Tiền tệ, số lượng, ngày giờ $\rightarrow$ `tien_vnd`, `so_luong`, `ngay_thang`, `thoi_gian`
4. **Loại bỏ ký tự đặc biệt**: Giữ lại các chữ cái tiếng Việt có dấu và số.
5. **Lọc từ dừng (Stopwords)**: Loại bỏ các từ không mang ý nghĩa phân biệt spam/ham (`là`, `và`, `của`, `các`, `được`...).

### Bước 2: Trích xuất đặc trưng TF-IDF (`src/features.py`)
* Phân chia tập dữ liệu: **80% Train (2.392 mẫu)** và **20% Test (598 mẫu)** phân tầng (`stratify=y`, `random_state=42`).
* Cấu hình TF-IDF:
  * `ngram_range=(1, 2)`: Khai thác cả từ đơn (Unigram) và cụm 2 từ (Bigram tiếng Việt).
  * `max_features=3000`: Giới hạn 3.000 đặc trưng mang giá trị thông tin cao nhất.
  * `sublinear_tf=True`: Áp dụng tỷ lệ logarithmic $(1 + \log(TF))$ để giảm sự chênh lệch do lặp từ.
* Tuyệt đối không rò rỉ dữ liệu: Vectorizer chỉ thực hiện `fit_transform` trên tập Train và `transform` trên tập Test.

---

## 5. Kết quả đối chuẩn 10 thuật toán Machine Learning

Thực nghiệm được thực hiện độc lập trên tập kiểm thử 598 mẫu (438 tin Ham và 160 tin Spam):

| STT | Thuật toán (Algorithm) | Accuracy | **Precision** | Recall | F1-Score | Số lỗi báo động nhầm (FP) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Extra Trees Classifier (ETC)** | **97.32%** | **98.00%** | 91.88% | 94.84% | **3** |
| 2 | **Random Forest (RF)** | **97.49%** | **96.18%** | 94.38% | 95.27% | 6 |
| 3 | **Support Vector Classifier (SVC)** | **97.16%** | **94.41%** | 95.00% | 94.70% | 9 |
| 4 | **Logistic Regression (LR)** | 95.82% | 92.99% | 91.25% | 92.11% | 11 |
| 5 | **Gradient Boosting (GBDT)** | 94.98% | 92.76% | 88.12% | 90.38% | 11 |
| 6 | **XGBoost (xgb)** | 96.49% | 92.64% | 94.38% | 93.50% | 12 |
| 7 | **AdaBoost Classifier** | 93.48% | 85.38% | 91.25% | 88.22% | 25 |
| 8 | **Multinomial Naïve Bayes** | 92.47% | 84.85% | 87.50% | 86.15% | 25 |
| 9 | **Decision Tree (DT)** | 92.81% | 81.28% | 95.00% | 87.61% | 35 |
| 10 | **K-Nearest Neighbors (KN)** | 71.24% | 48.19% | 100.00% | 65.04% | 172 |

### Nhận xét & Phân tích chuyên môn (Slide L2, L3, L4, L5)
1. **Lý do lựa chọn Naïve Bayes để đóng gói & triển khai**:
   * Theo đúng phương pháp luận bài báo gốc IEEE 2023, Multinomial Naïve Bayes có tốc độ suy luận cực nhanh (dưới 1 mili-giây), dung lượng tệp mô hình siêu nhẹ (**96 KB**), không tiêu tốn RAM, phù hợp tuyệt đối cho môi trường ứng dụng di động và thiết bị biên.
2. **Hiệu năng của các mô hình Ensemble (Extra Trees, Random Forest, SVC)**:
   * Đạt hiệu năng phân loại cao nhất với Accuracy và Precision vượt trên 97%, số lượng tin nhắn bị báo động nhầm (False Positive) rất thấp.
3. **Hiện tượng giảm sút hiệu năng của KNN**:
   * Do đặc thù không gian vector hóa TF-IDF có số chiều lớn (3.000 chiều) và độ thưa thớt cao (98.37%), hiện tượng **lời nguyền số chiều (Curse of Dimensionality)** khiến khoảng cách hình học Euclidean trong KNN bị suy giảm ý nghĩa phân tách.

---

## 6. Thảo luận mở rộng theo bài báo khoa học

### Mở rộng bài toán sang Email Spam (Mục VI)
* **SMS Spam**: Dung lượng ngắn (thường dưới 160 ký tự), nhiều từ viết tắt, tiếng lóng, thiếu thông tin ngữ cảnh, đòi hỏi tiền xử lý chuyên sâu.
* **Email Spam**: Chiều dài văn bản lớn hơn, ngữ cảnh đầy đủ, có tiêu đề (Subject) và trường Header người gửi. Việc phát hiện thường dễ đạt độ chính xác cao hơn nhưng đòi hỏi không gian lưu trữ và thời gian trích xuất vector TF-IDF lâu hơn.

### Thách thức triển khai thực tế (Mục VII)
1. **Cơ chế bảo mật Sandbox trên hệ điều hành**:
   * Các ứng dụng nhắn tin bảo mật (iOS Messages, WhatsApp, Zalo) không cho phép bên thứ ba tự động đọc tin nhắn ngầm, người dùng cần chủ động kiểm tra nội dung nghi vấn.
2. **Kỹ thuật lách bộ lọc của kẻ phát tán Spam (Adversarial Evasion)**:
   * Spammer liên tục thay đổi cách gõ (ví dụ: `Fr33`, `V.C.B`, `l-i-n-k`, chèn khoảng trắng và ký tự ẩn) để làm sai lệch bộ tách từ TF-IDF. Hướng khắc phục trong tương lai là áp dụng kỹ thuật N-gram ký tự (Character-level N-gram) hoặc các mô hình ngôn ngữ ngữ cảnh như PhoBERT.

---

## 7. Lịch sử Commit Git dự án

Toàn bộ quá trình phát triển tuân thủ nghiêm ngặt quy chuẩn **Conventional Commits**:

| Task | Tên nhiệm vụ | Commit Hash | Mô tả commit |
|:---:|:---|:---:|:---|
| **Task 1** | Khởi tạo dự án & Dependencies | `9e5cc0a` | `chore: initialize project structure and dependencies` |
| **Task 2** | Pipeline tiền xử lý tiếng Việt | `2794cea` | `feat(nlp): implement vietnamese text cleaning pipeline` |
| **Task 3** | Khám phá dữ liệu EDA (Fig 3-6) | `42d204a` | `feat(eda): generate data distribution, heatmap and wordclouds (Fig 3-6)` |
| **Task 4** | Trích xuất đặc trưng TF-IDF | `d28663f` | `feat(features): extract tf-idf unigram and bigram features` |
| **Task 5** | Đối chuẩn 10 mô hình ML (Fig 7) | `1167c34` | `feat(models): train and benchmark 10 classifiers (Fig 7)` |
| **Task 6** | Đóng gói mô hình (.pkl) | `ecdacb5` | `feat(export): serialize naive bayes model and vectorizer` |
| **Task 7** | Giao diện Web Streamlit (Fig 8-10) | `7eb9348` | `refactor(ui): streamline app to focused live prediction interface without extra tabs` |
| **Task 8** | Hoàn thiện README & Tài liệu | `fcdd703` | `docs: complete readme and project documentation` |
