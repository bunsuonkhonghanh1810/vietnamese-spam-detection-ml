# RULES

Quy tắc phát triển dự án Phân loại tin nhắn rác tiếng Việt (Tái hiện bài báo IEEE ICSSIT 2023)
Môn học: Học máy cơ bản - Giảng viên: TS. Lương Thái Lê

---

## 1. Mục tiêu & Ý tưởng
- Tái hiện 100% phương pháp luận từ bài báo *Spam SMS (or) Email Detection and Classification using Machine Learning (IEEE ICSSIT 2023)* trên tập dữ liệu tiếng Việt (2.991 tin nhắn thực tế).
- Bám sát kiến thức trong các slide bài giảng:
  + Slide L2: Độ đo đánh giá (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
  + Slide L3: Cây quyết định (Decision Tree ID3/CART).
  + Slide L4: Naïve Bayes (Multinomial NB & Bernoulli NB, Laplace Smoothing).
  + Slide L5: Support Vector Machines (Linear & RBF Kernel), Boosting & Ensemble (Random Forest, AdaBoost).

---

## 2. Quy tắc dữ liệu & Khả năng tái lập (Reproducibility)
- Dữ liệu thô gốc `data/raw/vietnamese_sms_spam.csv` là bất biến (chỉ đọc, không ghi đè).
- Dữ liệu sau khi làm sạch lưu tại `data/processed/`.
- Luôn cố định `random_state=42` ở mọi bước phân chia Train/Test (tỉ lệ 80/20) và khởi tạo mô hình.

---

## 3. Tiêu chuẩn đánh giá mô hình (Evaluation Standard)
- Dữ liệu thực tế mất cân bằng nhãn (~73.3% Ham vs ~26.7% Spam).
- **Precision là chỉ số ưu tiên hàng đầu**: Tuyệt đối tránh lỗi *False Positive (báo động nhầm - chặn nhầm tin quan trọng của người dùng như mã OTP, tin công việc)*.
- Luôn báo cáo đầy đủ cả 4 chỉ số: Accuracy, Precision, Recall, F1-Score kèm ma trận nhầm lẫn (Confusion Matrix).

---

## 4. Quy tắc phân chia thư mục
- `data/`: `raw/` (dữ liệu gốc), `processed/` (dữ liệu sạch).
- `docs/`: Tài liệu hướng dẫn, kế hoạch tasks, rules.
- `paper/`: Bài báo khoa học gốc.
- `slide/`: Slide bài giảng môn học.
- `src/`: Toàn bộ mã nguồn Python theo module.
- `models/`: Mô hình đã huấn luyện (.pkl) và biểu đồ kết quả (.png, .csv).

---

## 5. Quy tắc Git Commit
- Áp dụng chuẩn **Conventional Commits**: `chore:`, `feat:`, `fix:`, `docs:`, `refactor:`.
- **Atomic Commit**: Thực hiện từng task độc lập, kiểm thử chạy thành công mới tiến hành commit.
