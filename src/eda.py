# -*- coding: utf-8 -*-
"""
eda.py
Module phân tích khám phá dữ liệu (Exploratory Data Analysis - EDA) & Trực quan hóa
Tái hiện trung thực các biểu đồ trong bài báo IEEE ICSSIT 2023:
- Fig 3: Biểu đồ tròn tỷ lệ nhãn Ham vs Spam (Target Distribution Pie Chart)
- Fig 4: Biểu đồ nhiệt tương quan đặc trưng bề mặt (Correlation Heatmap)
- Fig 5: Đám mây từ khóa tin nhắn Spam (Spam WordCloud)
- Fig 6: Đám mây từ khóa tin nhắn Ham (Ham WordCloud)
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Cấu hình encoding console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình font chữ hỗ trợ tiếng Việt cho matplotlib
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def count_sentences(text: str) -> int:
    """Đếm số câu trong tin nhắn dựa trên các dấu ngắt câu hoặc xuống dòng."""
    if not isinstance(text, str) or not text.strip():
        return 0
    sentences = re.split(r'[\n.!?]+', text)
    return max(1, len([s for s in sentences if s.strip()]))

def run_eda_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_csv = os.path.join(base_dir, "data", "processed", "vietnamese_sms_spam_cleaned.csv")
    fig_dir = os.path.join(base_dir, "models", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print(" BẮT ĐẦU QUY TRÌNH PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) - TASK 3")
    print("=" * 70)

    print(f"-> Đang đọc dữ liệu sạch: {processed_csv}")
    df = pd.read_csv(processed_csv)
    print(f"-> Tổng số bản ghi: {len(df)}")

    # 1. Trích xuất các thuộc tính bề mặt phục vụ bài báo
    print("-> Đang trích xuất thuộc tính: số ký tự, số từ, số câu...")
    df["num_characters"] = df["message"].astype(str).apply(len)
    df["num_words"] = df["message"].astype(str).apply(lambda x: len(x.split()))
    df["num_sentences"] = df["message"].astype(str).apply(count_sentences)

    # Thống kê mô tả
    print("\n--- BẢNG THỐNG KÊ MÔ TẢ THEO NHÃN ---")
    stats = df.groupby("label")[["num_characters", "num_words", "num_sentences"]].describe()
    print(stats)

    ham_mean_chars = df[df["label"] == 0]["num_characters"].mean()
    spam_mean_chars = df[df["label"] == 1]["num_characters"].mean()
    print(f"\n* Chiều dài ký tự trung bình: Ham = {ham_mean_chars:.1f} ký tự | Spam = {spam_mean_chars:.1f} ký tự")
    print("  => Tin nhắn Spam có xu hướng dài hơn đáng kể so với tin nhắn thông thường!")

    # -------------------------------------------------------------
    # FIGURE 3: Biểu đồ tròn tỷ lệ nhãn (Fig 3. Dataset classifying spam and ham)
    # -------------------------------------------------------------
    print("\n[1/4] Đang tạo Figure 3: Biểu đồ tròn tỷ lệ nhãn (chuẩn flat, không cắt lát)...")
    plt.figure(figsize=(7, 7))
    label_counts = df["label"].value_counts().sort_index()
    labels = ["Ham (Không phải rác)", "Spam (Tin rác)"]
    colors = ["#4A90E2", "#FF6B6B"]

    plt.pie(
        label_counts,
        labels=labels,
        colors=colors,
        autopct="%1.2f%%",
        startangle=90,
        textprops={"fontsize": 12, "weight": "bold"}
    )
    plt.title("Fig 3. Dataset classifying spam and ham", fontsize=14, weight="bold", pad=20)
    fig3_path = os.path.join(fig_dir, "fig3_target_distribution_pie.png")
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f" -> Đã lưu Fig 3 tại: {fig3_path}")

    # -------------------------------------------------------------
    # FIGURE 4: Biểu đồ nhiệt tương quan (Correlation Heatmap)
    # -------------------------------------------------------------
    print("\n[2/4] Đang tạo Figure 4: Biểu đồ nhiệt tương quan (Correlation Heatmap)...")
    plt.figure(figsize=(8, 6))
    corr_cols = ["label", "num_characters", "num_words", "num_sentences"]
    corr_matrix = df[corr_cols].corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        linewidths=1,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Figure 4: Correlation Heatmap of Structural Features with Label", fontsize=13, weight="bold", pad=15)
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10, rotation=0)
    fig4_path = os.path.join(fig_dir, "fig4_correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f" -> Đã lưu Fig 4 tại: {fig4_path}")
    print(f" -> Hệ số tương quan giữa nhãn Spam và số ký tự: {corr_matrix.loc['label', 'num_characters']:.2f}")

    # -------------------------------------------------------------
    # FIGURE 5: Đám mây từ khóa Spam (Spam WordCloud)
    # -------------------------------------------------------------
    print("\n[3/4] Đang tạo Figure 5: Đám mây từ khóa Spam (Spam WordCloud)...")
    spam_corpus = " ".join(df[df["label"] == 1]["cleaned_message"].dropna())
    
    spam_wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="Reds",
        max_words=80,
        random_state=42,
        collocations=False
    ).generate(spam_corpus)

    plt.figure(figsize=(12, 6))
    plt.imshow(spam_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Figure 5: Word Cloud of Vietnamese Spam Messages", fontsize=15, weight="bold", pad=15)
    fig5_path = os.path.join(fig_dir, "fig5_spam_wordcloud.png")
    plt.tight_layout()
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f" -> Đã lưu Fig 5 tại: {fig5_path}")

    # -------------------------------------------------------------
    # FIGURE 6: Đám mây từ khóa Ham (Ham WordCloud)
    # -------------------------------------------------------------
    print("\n[4/4] Đang tạo Figure 6: Đám mây từ khóa Ham (Ham WordCloud)...")
    ham_corpus = " ".join(df[df["label"] == 0]["cleaned_message"].dropna())
    
    ham_wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="Blues",
        max_words=80,
        random_state=42,
        collocations=False
    ).generate(ham_corpus)

    plt.figure(figsize=(12, 6))
    plt.imshow(ham_wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Figure 6: Word Cloud of Vietnamese Ham Messages", fontsize=15, weight="bold", pad=15)
    fig6_path = os.path.join(fig_dir, "fig6_ham_wordcloud.png")
    plt.tight_layout()
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f" -> Đã lưu Fig 6 tại: {fig6_path}")

    print("\n" + "=" * 70)
    print(" HOÀN TẤT TASK 3! Toàn bộ 4 biểu đồ Fig 3 - Fig 6 đã được lưu trữ chuẩn.")
    print("=" * 70)

if __name__ == "__main__":
    run_eda_pipeline()
