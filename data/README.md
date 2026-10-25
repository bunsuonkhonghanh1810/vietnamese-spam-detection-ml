# BỘ DỮ LIỆU TIN NHẮN RÁC & LỪA ĐẢO TIẾNG VIỆT (VIETNAMESE SMS SPAM/PHISHING DATASET)

## 1. Nguồn gốc & Giới thiệu
* **Tên tệp**: `vietnamese_sms_spam.csv`
* **Nguồn**: Dự án nghiên cứu an ninh mạng và tin nhắn lừa đảo tại Việt Nam (*Quality-Assured Vietnamese SMS Phishing Dataset*).
* **Mục tiêu**: Phục vụ bài toán học máy phân loại văn bản tiếng Việt phát hiện tin nhắn lừa đảo, tin nhắn rác (Spam) và tin nhắn thông thường (Ham).
* **Năm thu thập**: 2026 (dữ liệu thực tế, phản ánh đúng các hình thức lừa đảo qua tin nhắn viễn thông hiện hành tại Việt Nam).

## 2. Thông số kỹ thuật
* **Tổng số mẫu**: 2.991 tin nhắn.
* **Định dạng**: File CSV, bảng mã UTF-8.
* **Cấu trúc cột**:
  1. `message_id`: Mã định danh tin nhắn (ví dụ: `COM_001`, `TE_027`...).
  2. `date`: Ngày thu thập / ghi nhận tin nhắn.
  3. `message`: Toàn bộ nội dung văn bản của tin nhắn SMS.
  4. `label`: Nhãn phân loại nhị phân:
     * `0`: **Ham** (Tin nhắn hợp lệ: mã OTP, biến động số dư ngân hàng, thông báo cước nhà mạng Viettel/Vina/Mobi, tin nhắn công việc).
     * `1`: **Spam** (Tin nhắn rác / lừa đảo: giả mạo ngân hàng Vietcombank/ACB/Vietinbank, lừa đảo nạp tiền, việc làm online, cờ bạc, link độc hại).

## 3. Phân bố nhãn (Class Distribution)
* **Ham (`0`)**: 2.193 tin nhắn (tỉ lệ **73.32%**)
* **Spam (`1`)**: 798 tin nhắn (tỉ lệ **26.68%**)
* **Đặc điểm**: Dữ liệu có tính mất cân bằng tự nhiên (Imbalanced Dataset), phản ánh đúng thực tế khi tin nhắn bình thường luôn chiếm đa số so với tin nhắn rác.

## 4. Bảo mật & Ẩn danh (PII Anonymization)
Các thông tin nhạy cảm của người dùng thực tế đã được chuẩn hóa thành các token chung:
* `[MONEY]`: Số tiền giao dịch.
* `[TIME]`: Thời gian, khung giờ.
* `[DATE]`: Ngày tháng năm.
* `[NUMBER]`: Các dãy số định danh cá nhân / số điện thoại.
