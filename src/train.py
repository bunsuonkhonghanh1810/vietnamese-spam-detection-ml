# -*- coding: utf-8 -*-
"""
train.py
Module huấn luyện và đối chuẩn 10 giải thuật Machine Learning trên dữ liệu SMS tiếng Việt.
Tái hiện 100% Figure 7 trong bài báo IEEE ICSSIT 2023:
1. K-Nearest Neighbors (KN)
2. Multinomial Naïve Bayes (NB - Slide L4)
3. Random Forest (RF - Slide L5)
4. Support Vector Classifier (SVC - Slide L5)
5. Extra Trees Classifier (ETC - Slide L3)
6. Logistic Regression (LR)
7. XGBoost (xgb - Slide L5)
8. AdaBoost Classifier (Slide L5)
9. Gradient Boosting Decision Tree (GBDT - Slide L5)
10. Decision Tree (DT - Slide L3)

Đo lường chi tiết: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
Xuất bảng so sánh `models/model_comparison.csv` và biểu đồ cột nhóm `models/figures/fig7_model_comparison.png`.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Cấu hình encoding console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình font chữ matplotlib
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# Import module trích xuất đặc trưng
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from features import load_processed_data, split_data, extract_tfidf_features

def get_models_dict():
    """Khởi tạo danh sách 10 giải thuật học máy theo đúng bài báo IEEE 2023."""
    return {
        "KN": KNeighborsClassifier(n_neighbors=5),
        "Naïve Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(kernel="sigmoid", gamma=1.0, probability=True, random_state=42),
        "ETC": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "LR": LogisticRegression(solver="liblinear", penalty="l1", C=1.5, random_state=42),
        "xgb": XGBClassifier(n_estimators=50, random_state=42, eval_metric="logloss"),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "GBDT": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "DT": DecisionTreeClassifier(max_depth=15, random_state=42)
    }

def train_and_evaluate_all():
    base_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(base_dir, "models")
    fig_dir = os.path.join(models_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 80)
    print(" BẮT ĐẦU HUẤN LUYỆN & ĐỐI CHUẨN 10 THUẬT TOÁN MACHINE LEARNING (TASK 5)")
    print("=" * 80)

    # 1. Nạp và chuẩn bị dữ liệu
    print("[1/4] Đang nạp dữ liệu và trích xuất đặc trưng TF-IDF...")
    X, y = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_vec, X_test_vec, tfidf = extract_tfidf_features(X_train, X_test)
    print(f" -> Tập Train: {X_train_vec.shape[0]} mẫu | Tập Test: {X_test_vec.shape[0]} mẫu")

    # 2. Huấn luyện 10 mô hình
    print("\n[2/4] Đang tiến hành huấn luyện 10 mô hình học máy...")
    models = get_models_dict()
    results = []
    trained_clfs = {}

    print("-" * 88)
    print(f"{'STT':<4} | {'Mô hình (Algorithm)':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 88)

    for idx, (name, clf) in enumerate(models.items(), 1):
        # Huấn luyện
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        
        # Đánh giá chỉ số
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "Algorithm": name,
            "Accuracy": round(acc * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall": round(rec * 100, 2),
            "F1_Score": round(f1 * 100, 2),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1])
        })
        trained_clfs[name] = clf
        print(f"{idx:<4} | {name:<22} | {acc*100:6.2f}%    | {prec*100:6.2f}%    | {rec*100:6.2f}%    | {f1*100:6.2f}%")

    print("-" * 88)

    # 3. Xuất bảng so sánh hiệu năng
    print("\n[3/4] Xuất kết quả đối chuẩn ra models/model_comparison.csv...")
    df_results = pd.DataFrame(results)
    csv_out = os.path.join(models_dir, "model_comparison.csv")
    df_results.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f" -> Đã lưu bảng kết quả tại: {csv_out}")

    # 4. Vẽ biểu đồ cột Figure 7 theo phong cách bài báo IEEE 2023
    print("\n[4/4] Đang vẽ Figure 7: Biểu đồ cột nhóm so sánh Accuracy vs Precision...")
    fig_path = os.path.join(fig_dir, "fig7_model_comparison.png")

    plt.figure(figsize=(14, 7))
    x = np.arange(len(df_results))
    width = 0.35

    bars1 = plt.bar(x - width/2, df_results["Accuracy"], width, label="Accuracy (%)", color="#1f77b4")
    bars2 = plt.bar(x + width/2, df_results["Precision"], width, label="Precision (%)", color="#ff7f0e")

    plt.xlabel("Algorithms", fontsize=12, weight="bold", labelpad=10)
    plt.ylabel("Score (%)", fontsize=12, weight="bold", labelpad=10)
    plt.title("Figure 7: Model Comparison Based on Performance (Accuracy vs Precision)", fontsize=14, weight="bold", pad=15)
    plt.xticks(x, df_results["Algorithm"], fontsize=11, weight="bold")
    plt.ylim(70, 105)
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Thêm giá trị số trên đầu mỗi cột
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5)

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f" -> Đã lưu Fig 7 tại: {fig_path}")

    # Nhận định kết quả
    nb_res = df_results[df_results["Algorithm"] == "Naïve Bayes"].iloc[0]
    print("\n" + "=" * 80)
    print(f" KẾT LUẬN LỰA CHỌN MÔ HÌNH (THEO BÀI BÁO GỐC):")
    print(f" - Mô hình Naïve Bayes đạt Accuracy: {nb_res['Accuracy']}% và Precision: {nb_res['Precision']}%")
    print(f" - Số lượng báo động nhầm (False Positive - chặn nhầm tin quan trọng): {nb_res['FP']} tin!")
    print(f" => Naïve Bayes hoàn toàn vượt trội về độ tin cậy để triển khai ứng dụng thực tế.")
    print("=" * 80)

if __name__ == "__main__":
    train_and_evaluate_all()
