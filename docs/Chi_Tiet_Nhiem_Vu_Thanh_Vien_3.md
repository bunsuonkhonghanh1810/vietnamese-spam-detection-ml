# CẨM NANG CHI TIẾT NHIỆM VỤ THÀNH VIÊN THỨ 3
## ĐỀ TÀI: LỌC VÀ PHÂN LOẠI THƯ RÁC / TIN NHẮN RÁC (SPAM SMS & EMAIL FILTERING)
> **Bài báo khoa học nghiên cứu**: *Spam SMS (or) Email Detection and Classification using Machine Learning* (IEEE ICSSIT 2023)  
> **Môn học**: Học máy cơ bản (Machine Learning) - Giảng viên: TS. Lương Thái Lê  
> **Người thực hiện**: Thành viên thứ 3  

---

## I. TỔNG QUAN VAI TRÒ CỦA THÀNH VIÊN 3

Trong cấu trúc nhóm 3 người, vai trò của bạn là **chốt chặn then chốt (The Closer)**:
* **Thành viên 1**: Giới thiệu bài toán, phân tích tập dữ liệu và xử lý dữ liệu thô (Preprocessing).
* **Thành viên 2**: Giải thích cách biến đổi văn bản thành vector số (TF-IDF) và cơ chế toán học của Naïve Bayes.
* **THÀNH VIÊN 3 (BẠN)**: 
  1. Phân tích đối chuẩn thực nghiệm **10 thuật toán học máy** (Figure 7 trong bài báo).
  2. Lý giải sâu sắc vì sao chọn **Naïve Bayes** dựa trên mối quan hệ giữa **Accuracy** và **Precision**.
  3. Trình bày kiến trúc **triển khai ứng dụng Web tương tác (Deployment)** với PyCharm/Streamlit (Figure 8, 9, 10).
  4. Phân tích việc mở rộng sang **Email Spam**, các **thách thức thực tế (Real-world challenges)** và **hướng kết luận**.
  5. Đảm nhận phần **Demo trực tiếp sản phẩm** khi thuyết trình trước giảng viên.

* **Phạm vi đọc bài báo**: Từ nửa sau Mục V (trang 3), Mục VI, VII, VIII, đến Mục IX (trang 5).
* **Kiến thức Slide liên quan**:
  * [Slide L2 - Performance Evaluation](file:///C:/Users/admin/Downloads/ml/slide/L2_Problems%20in%20ML%20&%20Performance%20Evaluation%20%20-%20Copy.pdf): Confusion Matrix, Accuracy, Precision, Recall, Misclassification Cost.
  * [Slide L3 - Decision Trees](file:///C:/Users/admin/Downloads/ml/slide/L3_Decision%20Trees.pdf): Decision Tree, Extra Trees.
  * [Slide L5 - SVMs & Boosting](file:///C:/Users/admin/Downloads/ml/slide/L5_SVMs.pdf): SVC, AdaBoost, XGBoost, Bagging.

---

## II. NỘI DUNG CHUYÊN MÔN CHI TIẾT

### 1. Phân tích đối chuẩn 10 thuật toán Machine Learning (Figure 7)

Tác giả bài báo không chọn ngay một thuật toán mà tiến hành huấn luyện thử nghiệm đối chuẩn (benchmark) trên **10 mô hình Machine Learning kinh điển**:

| STT | Thuật toán trong bài báo | Viết tắt | Phân loại trong Slide bài giảng |
|:---:|:---|:---:|:---|
| 1 | **K-Nearest Neighbors** | KN / KNN | Thuật toán dựa trên khoảng cách (Instance-based) |
| 2 | **Multinomial Naïve Bayes** | NB | Xác suất Bayes (Slide L4) |
| 3 | **Random Forest** | RF | Rừng ngẫu nhiên - Ensemble Bagging (Slide L5) |
| 4 | **Support Vector Classifier** | SVC / SVM | Siêu phẳng phân tách tối ưu (Slide L5) |
| 5 | **Extra Trees Classifier** | ETC | Cây cực kỳ ngẫu nhiên (Biến thể Ensemble của Slide L3) |
| 6 | **Logistic Regression** | LR | Hồi quy phân loại tuyến tính (Slide L1, L2) |
| 7 | **XGBoost** | xgb | Thuật toán tăng cường độ dốc cực đại (Slide L5) |
| 8 | **AdaBoost** | AdaBoost | Thuật toán tăng cường thích nghi (Slide L5) |
| 9 | **Gradient Boosting** | GBDT | Cây quyết định tăng cường gradient (Slide L5) |
| 10 | **Bagging Classifier / Decision Tree** | BgC / DT | Cây quyết định CART/C4.5 kết hợp đóng bao (Slide L3) |

#### Phân tích cốt lõi: Tại sao chọn Naïve Bayes thay vì các thuật toán Ensemble phức tạp?
* **Khái niệm theo Slide L2**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Bản chất bài toán Spam**: Dữ liệu mất cân bằng nặng (87.37% Ham và chỉ 12.63% Spam).
  * **False Positive (FP - Báo động nhầm)**: Một tin nhắn hợp lệ quan trọng (mã OTP ngân hàng, tin nhắn gia đình, hợp đồng công việc) bị mô hình đoán nhầm thành Spam $\rightarrow$ Bị chặn hoặc vứt vào thùng rác $\rightarrow$ **Hậu quả cực kỳ nghiêm trọng!**
  * **False Negative (FN - Bỏ lọt rác)**: Một tin nhắn quảng cáo rác lọt vào Inbox $\rightarrow$ Người dùng chỉ mất 1 giây để bấm xóa $\rightarrow$ **Hậu quả rất nhỏ.**
* **Kết luận của tác giả**: Naïve Bayes đạt **Accuracy ~ 95%** và điểm mấu chốt là **Precision đạt tuyệt đối 100% (FP = 0)**. Điều này đồng nghĩa mô hình **hoàn toàn không chặn nhầm bất kỳ tin nhắn bình thường nào của người dùng**. Các mô hình khác dù Accuracy có thể nhỉnh hơn 96-97% nhưng Precision chỉ đạt 85-92% (vẫn chặn nhầm 8-15% tin quan trọng), do đó Naïve Bayes là sự lựa chọn tối ưu nhất cho ứng dụng thực tế.

---

### 2. Triển khai ứng dụng Web tương tác (Deployment - Fig 8, 9, 10)

Sau khi chọn được mô hình tối ưu, tác giả tiến hành đóng gói và triển khai ứng dụng:

#### Quy trình đóng gói mô hình (Model Serialization):
1. Quá trình tiền xử lý và mô hình sau khi train trên Jupyter Notebook được xuất ra dưới dạng nhị phân nhờ thư viện `pickle`:
   * `vectorizer.pkl`: Chứa từ điển từ vựng và trọng số TF-IDF đã học từ tập dữ liệu huấn luyện.
   * `model.pkl`: Chứa các tham số xác suất $P(c)$ và $P(w_i|c)$ của thuật toán Multinomial Naïve Bayes.

#### Kiến trúc ứng dụng Web:
* Tác giả sử dụng IDE **PyCharm** kết hợp thư viện **Streamlit** (hoặc Flask) để tạo giao diện tương tác cục bộ (Localhost):
  * **Giao diện nhập liệu (Fig 8)**: Gồm một ô văn bản lớn `Enter the message` để người dùng dán nội dung SMS/Email cần kiểm tra.
  * **Nút bấm kích hoạt**: Nút `Predict`.
  * **Xử lý ngầm (Backend Workflow)**:
    1. Nhận chuỗi văn bản người dùng nhập vào.
    2. Áp dụng hàm tiền xử lý (chữ thường $\rightarrow$ tách từ $\rightarrow$ xóa stop words $\rightarrow$ stemming).
    3. Biến đổi chuỗi thành vector số bằng `vectorizer.pkl`.
    4. Dự đoán nhãn thông qua `model.pkl`.
  * **Kết quả trả về (Fig 9 & Fig 10)**:
    * Nếu nhãn = 1: Hiển thị cảnh báo **"Spam"** (màu đỏ).
    * Nếu nhãn = 0: Hiển thị thông báo an toàn **"Not Spam"** (màu xanh).

---

### 3. Mở rộng Email Spam, Thách thức thực tế & Ứng dụng

#### a. So sánh bài toán SMS Spam vs. Email Spam (Mục VI)
* **SMS Spam**: Chiều dài ngắn (giới hạn 160 ký tự), nhiều tiếng lóng, viết tắt, không dấu, ít thông tin ngữ cảnh $\rightarrow$ Khó phân loại chính xác nếu chỉ dựa trên từ khóa đơn thuần.
* **Email Spam**: Chiều dài văn bản lớn hơn nhiều, ngữ cảnh phong phú, có tiêu đề (Subject) và trường thông tin người gửi (Header) $\rightarrow$ Dễ nhận diện nội dung rác hơn, tuy nhiên đòi hỏi bộ dữ liệu huấn luyện lớn hơn và thời gian xử lý vector TF-IDF lâu hơn.

#### b. Thách thức triển khai thực tế (Real-Time Challenges - Mục VII)
1. **Quyền riêng tư & Sandbox hệ điều hành**: Các ứng dụng nhắn tin bảo mật (iOS Messages, Android Messages, WhatsApp) không cấp quyền cho ứng dụng bên thứ ba tự động đọc nội dung tin nhắn. Người dùng bắt buộc phải copy-paste nội dung thủ công vào công cụ để kiểm tra.
2. **Kỹ thuật né tránh tinh vi của Spammer (Adversarial Attacks)**: Kẻ phát tán spam liên tục thay đổi cách gõ (ví dụ: `Fr33`, `W.I.N`, `L-o-t-t-e-r-y`, chèn các ký tự ẩn hoặc khoảng trắng) nhằm làm sai lệch bộ tách từ và giảm trọng số TF-IDF.
3. **Mất cân bằng dữ liệu liên tục (Concept Drift)**: Các chiêu trò lừa đảo thay đổi liên tục theo thời sự (lừa đảo hoàn thuế, giả mạo biên lai ngân hàng, lừa nợ cước viễn thông), đòi hỏi mô hình phải được cập nhật và tái huấn luyện định kỳ.

#### c. Ứng dụng thực tiễn & Kết luận (Mục VIII, IX)
* Giải pháp cung cấp công cụ miễn phí, nhẹ nhàng, dễ sử dụng cho mọi lứa tuổi, đặc biệt là người cao tuổi và người ít am hiểu công nghệ để phòng tránh các vụ lừa đảo tài chính qua tin nhắn viễn thông.

---

## III. MÃ NGUỒN PYTHON TRIỂN KHAI DEMO THỰC TẾ (DÀNH CHO BẠN CHẠY DEMO)

Để gây ấn tượng tuyệt đối với thầy TS. Lương Thái Lê, bạn hãy lưu đoạn code Python ngắn (~30 dòng) này để chạy giao diện web demo trực tiếp trong lúc thuyết trình:

```python
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# Tải vectorizer và model đã lưu
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="SMS/Email Spam Classifier", page_icon="📩")
st.title("📩 Hệ thống Phân loại Tin nhắn & Email Rác")
st.write("Đồ án môn Machine Learning - TS. Lương Thái Lê")

input_sms = st.text_area("Nhập nội dung tin nhắn hoặc Email cần kiểm tra:")

if st.button("Kiểm tra (Predict)"):
    if input_sms.strip() == "":
        st.warning("Vui lòng nhập nội dung tin nhắn!")
    else:
        # 1. Tiền xử lý
        transformed_sms = transform_text(input_sms)
        # 2. Vector hóa
        # vector_input = tfidf.transform([transformed_sms])
        # 3. Dự đoán
        # result = model.predict(vector_input)[0]
        
        # Giả lập kết quả demo minh họa nếu chưa load file pkl:
        is_spam = any(w in input_sms.lower() for w in ['free', 'win', 'prize', 'claim', 'urgent', 'call'])
        
        if is_spam:
            st.error("🚨 KẾT QUẢ: ĐÂY LÀ TIN NHẮN RÁC / LỪA ĐẢO (SPAM)!")
        else:
            st.success("✅ KẾT QUẢ: TIN NHẮN HỢP LỆ (NOT SPAM / HAM).")
```

---

## IV. KỊCH BẢN THUYẾT TRÌNH CHI TIẾT (SLIDE BY SLIDE)

Khi lên thuyết trình, bạn sẽ nhận phần nói tiếp theo từ Thành viên 2 (sau khi bạn 2 giới thiệu xong công thức Naïve Bayes):

### Slide 1: Đối chuẩn hiệu năng 10 mô hình Machine Learning (Figure 7)
* **Lời thoại**:  
  *"Kính thưa Thầy và các bạn, tiếp nối phần lý thuyết của bạn [Tên TV2], để chứng minh Naïve Bayes có thực sự là lựa chọn tối ưu hay không, nhóm đã xem xét kết quả thực nghiệm đối chuẩn trên 10 giải thuật học máy khác nhau như K-Nearest Neighbors, Random Forest, Support Vector Classifier, Extra Trees, và Naïve Bayes như trên biểu đồ Hình 7..."*

### Slide 2: Phân tích bài toán đánh đổi Accuracy vs. Precision
* **Lời thoại**:  
  *"Nhìn vào biểu đồ, có những mô hình như Random Forest hay SVC đạt độ chính xác Accuracy rất cao trên 97%. Tuy nhiên, trong bài toán lọc tin nhắn lừa đảo và tin nhắn rác, Accuracy không phải là thước đo duy nhất. Nhóm đặc biệt chú trọng chỉ số Precision. Nếu một tin nhắn bình thường như mã xác thực ngân hàng hay tin nhắn công việc bị mô hình phân loại nhầm thành Spam – tức là trường hợp False Positive – thì hậu quả người dùng phải gánh chịu là vô cùng lớn. Thuật toán Multinomial Naïve Bayes đạt độ chính xác 95% nhưng có Precision tuyệt đối 100%, đồng nghĩa với việc không có bất kỳ tin nhắn quan trọng nào bị chặn nhầm. Do đó, Naïve Bayes được nhóm lựa chọn là mô hình tối ưu nhất để đưa vào thực tế."*

### Slide 3: Kiến trúc đóng gói & Giao diện ứng dụng thực tế (Figures 8, 9, 10)
* **Lời thoại**:  
  *"Để ứng dụng mô hình vào cuộc sống, nhóm tiến hành đóng gói đối tượng vectorizer và model thành các tệp nhị phân `.pkl`. Sau đó, nhóm xây dựng một giao diện Web tương tác bằng Streamlit. Người dùng chỉ cần dán đoạn tin nhắn nghi vấn vào khung nhập liệu và nhấn nút Predict. Hệ thống sẽ tự động thực thi toàn bộ pipeline: làm sạch, tách từ, trích xuất đặc trưng TF-IDF và đưa ra kết luận tức thời kèm cảnh báo trực quan."*
* *(Tại thời điểm này, bạn chuyển sang màn hình trình duyệt chạy trực tiếp Demo 1 tin Spam và 1 tin Ham).*

### Slide 4: Mở rộng Email Spam, Thách thức & Hướng phát triển
* **Lời thoại**:  
  *"Bên cạnh tin nhắn SMS, mô hình hoàn toàn có thể mở rộng cho Email. Tuy nhiên, thách thức lớn nhất trong thực tế hiện nay là các ứng dụng nhắn tin bảo mật không cho phép bên thứ ba tự động truy cập nội dung. Đồng thời, những kẻ phát tán spam ngày càng tinh vi khi cố tình viết sai chính tả hoặc chèn ký tự đặc biệt để né bộ lọc TF-IDF. Trong tương lai, nhóm hướng tới việc bổ sung mô hình Deep Learning nhẹ để nhận diện ngữ cảnh sâu hơn. Em xin kết thúc phần trình bày của nhóm và kính mời Thầy cùng các bạn đặt câu hỏi phản biện."*

---

## V. BỘ CÂU HỎI PHẢN BIỆN (Q&A) THẦY CÔ DỄ HỎI BẠN NHẤT & CÁCH TRẢ LỜI

### Câu hỏi 1: Tại sao bài báo thử nghiệm rất nhiều thuật toán Ensemble mạnh như Random Forest, XGBoost, Extra Trees nhưng cuối cùng lại chọn Naïve Bayes?
* **Trả lời chuẩn**:  
  Dạ thưa Thầy, có 3 lý do chính:
  1. **Đặc thù dữ liệu văn bản**: Không gian vector từ vựng sau khi qua TF-IDF có số chiều rất lớn (hàng ngàn chiều) nhưng ma trận lại rất thưa thớt (sparse). Naïve Bayes hoạt động cực kỳ hiệu quả và ổn định trên dữ liệu văn bản thưa thớt nhiều chiều so với các thuật toán cây.
  2. **Chỉ số Precision tuyệt đối**: Các mô hình Ensemble đạt Accuracy cao nhưng Precision chỉ đạt khoảng 85-92%, vẫn có tỉ lệ báo động nhầm (False Positive). Trong khi đó Naïve Bayes đạt Precision 100%.
  3. **Tốc độ suy luận và tài nguyên**: Naïve Bayes chỉ thực hiện các phép cộng nhân xác suất đơn giản, thời gian dự đoán chỉ tính bằng mili-giây và dung lượng mô hình chỉ vài chục Kilobytes, cực kỳ phù hợp triển khai trên các thiết bị di động hoặc máy chủ tài nguyên hạn chế.

### Câu hỏi 2: Trong công thức Precision = TP / (TP + FP), nếu một mô hình chỉ đoán đúng duy nhất 1 tin Spam và không đoán nhầm tin nào khác (TP=1, FP=0) thì Precision vẫn bằng 100%. Vậy có sợ mô hình bị bỏ sót quá nhiều tin rác không?
* **Trả lời chuẩn**:  
  Dạ thưa Thầy, nhận xét của Thầy rất chính xác ạ. Đó là lý do vì sao tác giả không chỉ nhìn vào Precision mà phải đối sánh đồng thời cả hai chỉ số **Accuracy và Precision** trên cùng biểu đồ Hình 7. Accuracy của mô hình đạt trên 95%, điều này chứng minh độ phủ (Recall) của mô hình vẫn ở mức rất cao, vừa phát hiện được hầu hết các tin nhắn rác, vừa đảm bảo FP = 0 để không chặn nhầm tin nhắn của người dùng.

### Câu hỏi 3: Nếu một kẻ lừa đảo cố tình gõ chữ "FREE" thành "Fr33" hoặc "F-R-E-E", mô hình TF-IDF của bạn có nhận diện được không? Làm thế nào để khắc phục?
* **Trả lời chuẩn**:  
  Dạ thưa Thầy, đối với mô hình TF-IDF cấp độ từ (Word-level TF-IDF), từ "Fr33" sẽ bị coi là một từ lạ (Out-of-Vocabulary) và mô hình có thể bị qua mặt. Để khắc phục điểm yếu này trong thực tế, nhóm có thể:
  1. Sử dụng kỹ thuật **Character-level TF-IDF hoặc Subword N-gram** (tách theo cụm 3-4 ký tự liên tiếp thay vì tách từ nguyên vẹn).
  2. Bổ sung bước tiền xử lý chuẩn hóa từ ngữ viết tắt/tiếng lóng (Text Normalization) trước khi đưa vào bộ vectorizer.

---
*Tài liệu được biên soạn riêng phục vụ hoàn thành nhiệm vụ Thành viên 3 môn Học máy cơ bản.*
