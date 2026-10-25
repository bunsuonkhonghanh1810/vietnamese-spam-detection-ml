# TASKS

Dự án: Phân loại tin nhắn rác tiếng Việt (Tái hiện bài báo IEEE ICSSIT 2023)

---

## Task 1: Khởi tạo dự án & cấu hình Git [ĐÃ HOÀN THÀNH - Commit: 9e5cc0a]
- **Mục tiêu**: Chuẩn hóa môi trường và .gitignore
- **Files**: `.gitignore`, `requirements.txt`, `docs/RULES.md`
- **Git Commit**:
  ```bash
  git add .gitignore requirements.txt docs/ data/ paper/ slide/
  git commit -m "chore: initialize project structure and dependencies"
  ```

---

## Task 2: Tiền xử lý văn bản tiếng Việt [ĐÃ HOÀN THÀNH - Commit: 2794cea]
- **Mục tiêu**: Chuẩn hóa Unicode NFC, xử lý link/sđt/tiền tệ, xóa stopwords tiếng Việt
- **Files**: `src/preprocess.py`, `data/processed/vietnamese_sms_spam_cleaned.csv`
- **Git Commit**:
  ```bash
  git add src/preprocess.py data/processed/ .gitignore docs/TASKS.md
  git commit -m "feat(nlp): implement vietnamese text cleaning pipeline"
  ```

---

## Task 3: Khám phá dữ liệu EDA (Tái hiện Fig 3, 4, 5, 6) [ĐÃ HOÀN THÀNH - Commit: 42d204a]
- **Mục tiêu**: Biểu đồ tròn tỷ lệ nhãn (Fig 3), Heatmap tương quan độ dài (Fig 4), WordCloud Spam vs Ham (Fig 5, 6)
- **Files**: `src/eda.py`, `models/figures/`
- **Git Commit**:
  ```bash
  git add src/eda.py models/figures/ docs/TASKS.md
  git commit -m "feat(eda): generate data distribution, heatmap and wordclouds (Fig 3-6)"
  ```

---

## Task 4: Trích xuất đặc trưng TF-IDF [ĐÃ HOÀN THÀNH - Commit: d28663f]
- **Mục tiêu**: Phân chia Train/Test (80/20, `random_state=42`), vector hóa TF-IDF (Unigram + Bigram)
- **Files**: `src/features.py`
- **Git Commit**:
  ```bash
  git add src/features.py docs/TASKS.md
  git commit -m "feat(features): extract tf-idf unigram and bigram features"
  ```

---

## Task 5: Huấn luyện & Đối chuẩn 10 mô hình ML (Tái hiện Fig 7)
- **Mục tiêu**: Huấn luyện 10 mô hình (KNN, Naive Bayes, Random Forest, SVC, Extra Trees, Logistic Regression, XGBoost, AdaBoost, Gradient Boost, Decision Tree). Vẽ biểu đồ so sánh Accuracy vs Precision (Fig 7)
- **Files**: `src/train.py`, `models/model_comparison.csv`, `models/figures/fig7_model_comparison.png`
- **Git Commit**:
  ```bash
  git add src/train.py models/model_comparison.csv models/figures/fig7_model_comparison.png
  git commit -m "feat(models): train and benchmark 10 classifiers (Fig 7)"
  ```

---

## Task 6: Đóng gói mô hình tối ưu
- **Mục tiêu**: Chọn Multinomial Naive Bayes (Precision 100%), lưu file nhị phân bằng pickle
- **Files**: `src/export_model.py`, `models/vectorizer.pkl`, `models/model.pkl`
- **Git Commit**:
  ```bash
  git add src/export_model.py models/*.pkl
  git commit -m "feat(export): serialize naive bayes model and vectorizer"
  ```

---

## Task 7: Triển khai giao diện Web Streamlit (Tái hiện Fig 8, 9, 10)
- **Mục tiêu**: Giao diện nhập tin nhắn + nút Predict + cảnh báo Spam (đỏ) / Ham (xanh)
- **Files**: `src/app.py`
- **Git Commit**:
  ```bash
  git add src/app.py
  git commit -m "feat(ui): build streamlit web interface for live spam detection (Fig 8-10)"
  ```

---

## Task 8: Hoàn thiện README & Báo cáo
- **Mục tiêu**: Hướng dẫn chạy dự án, thảo luận Email Spam & thách thức né bộ lọc (Mục VI, VII)
- **Files**: `README.md`
- **Git Commit**:
  ```bash
  git add README.md
  git commit -m "docs: complete readme and project documentation"
  ```
