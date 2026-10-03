# Bước 8: Phân tích kết quả

Benchmark chạy ở chế độ offline với dữ liệu trong `data/`. Token là ước lượng
heuristic của lab; `Response quality` cũng là điểm heuristic, không phải đánh
giá của người dùng hay LLM.

## Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 334 | 9.395 | 0,00 | 0,40 | 0 | 0 |
| Advanced | 603 | 17.689 | 0,89 | 0,77 | 340 | 0 |

Advanced nhớ tốt hơn vì trích xuất các thông tin ổn định từ lời khai và lưu
vào `User.md`, sau đó đọc lại hồ sơ khi câu hỏi được hỏi trong thread mới.
Baseline chỉ có lịch sử theo thread nên không thể recall xuyên session. Điểm
recall của Advanced chưa đạt tuyệt đối, cho thấy heuristic trích xuất/nhận
diện câu hỏi vẫn có giới hạn.

Ở hội thoại thông thường, Advanced tốn nhiều hơn: vừa đưa hồ sơ vào ngữ cảnh,
vừa lưu short-term history; phần này chưa đủ dài để compact. Vì thế cả prompt
tokens (17.689 so với 9.395) và agent tokens (603 so với 334) đều cao hơn.

## Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 85 | 20.775 | 0,00 | 0,40 | 0 | 0 |
| Advanced | 172 | 12.946 | 1,00 | 0,93 | 243 | 5 |

Khi thread dài, compact memory giữ summary và một số message gần nhất thay vì
mang toàn bộ lịch sử theo từng lượt. Advanced compact 5 lần; prompt tokens giảm
7.829 (khoảng 37,7%) so với Baseline, trong khi vẫn đạt recall đầy đủ trên bộ
câu hỏi stress. Agent tokens vẫn cao hơn do phản hồi offline của Advanced dài
hơn; lợi ích của compact thể hiện chủ yếu ở lượng prompt context được xử lý.

## Tăng trưởng và rủi ro của memory

Profile tăng 340 bytes ở benchmark thường và 243 bytes ở stress benchmark.
Đây là mức nhỏ với dữ liệu hiện tại, nhưng `User.md` có thể tăng theo số fact
được lưu. Fact trích xuất sai hoặc đã lỗi thời có thể làm câu trả lời sai;
thông tin cá nhân còn tạo rủi ro riêng tư. Cần giới hạn loại fact, hỗ trợ sửa
xóa/cập nhật, và cân nhắc chính sách lưu trữ/decay khi dùng thực tế. Summary
cũng có thể làm mất ngữ cảnh không được trích thành fact.

## Kết luận

Persistent memory cải thiện recall qua thread mới; compaction có ích rõ hơn ở
hội thoại dài, còn hội thoại ngắn có thể tốn thêm token do chi phí nạp hồ sơ.
Các số liệu trên dùng estimator và quality heuristic của bài lab nên phù hợp
để so sánh tương đối giữa hai agent, không đại diện cho token usage hoặc chất
lượng của một nhà cung cấp LLM cụ thể.
