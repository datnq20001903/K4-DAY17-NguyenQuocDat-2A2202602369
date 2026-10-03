# Phân tích bài memory systems

## CP3: cấu hình chung

`load_config()` xác định đường dẫn dùng chung. Mặc định repo root được tính từ
vị trí `src/config.py`, không phụ thuộc working directory. Khi truyền `base_dir`,
cả `data_dir` và `state_dir` đều nằm dưới root đó. Hàm tạo `state/` bằng
`mkdir(parents=True, exist_ok=True)` trước khi trả về config.

Giữ nguyên bảy trường của `LabConfig`. Main model và judge model là hai
`ProviderConfig` riêng biệt, tránh sửa một config làm thay đổi config còn lại.

### Quy ước môi trường

Đọc `.env` tại root đã chọn bằng `python-dotenv`; biến môi trường của process
ưu tiên hơn `.env`. Việc đọc file không ghi vào `os.environ`, nên nạp config
của root khác trong cùng process không kế thừa giá trị từ file trước.
Không có `.env` thì config mặc định nạp được cả khi chưa cài `python-dotenv`.

| Biến | Mặc định / ý nghĩa |
|---|---|
| `COMPACT_THRESHOLD_TOKENS` | `1000`, số nguyên dương |
| `COMPACT_KEEP_MESSAGES` | `6`, số message gần nhất giữ nguyên, số nguyên dương |
| `LLM_PROVIDER` | `openai` |
| `LLM_MODEL` | Theo provider, xem bảng bên dưới |
| `LLM_TEMPERATURE` | `0`, số hữu hạn không âm |
| `LLM_API_KEY` | Ghi đè API key riêng của provider |
| `LLM_BASE_URL` | Ghi đè base URL riêng của provider |
| `JUDGE_PROVIDER` | Cùng provider với main model |
| `JUDGE_MODEL` | Cùng model với main nếu cùng provider; nếu khác, dùng mặc định của provider judge |
| `JUDGE_TEMPERATURE` | `0`, độc lập với temperature của main |
| `JUDGE_API_KEY` | Ghi đè key của judge; nếu cùng provider, mặc định kế thừa key của main |
| `JUDGE_BASE_URL` | Ghi đè URL của judge; nếu cùng provider, mặc định kế thừa URL của main |

Nếu judge chọn provider khác, key và URL mặc định lấy từ biến riêng của provider
đó. Giá trị trống được xử lý như chưa cấu hình.

| Provider | Model mặc định | API key | Base URL |
|---|---|---|---|
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` | `OPENAI_BASE_URL`, tùy chọn |
| `custom` | `local-model` | `CUSTOM_API_KEY`, tùy chọn | `CUSTOM_BASE_URL`, bắt buộc khi dựng model live |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY`, dự phòng `GOOGLE_API_KEY` | `GEMINI_BASE_URL`, tùy chọn |
| `anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL`, tùy chọn |
| `ollama` | `llama3.2` | Không cần | `OLLAMA_BASE_URL`, mặc định `http://localhost:11434` |
| `openrouter` | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL`, mặc định `https://openrouter.ai/api/v1` |

Tên model có thể ghi đè bằng `LLM_MODEL` / `JUDGE_MODEL`; đổi tên không tác động
logic memory. Với custom và Ollama, đặt tên đúng model mà server local phục vụ.

### Compact và offline

Ngưỡng 1000 token và giữ 6 message là điểm bắt đầu: dữ liệu standard có khoảng
476–756 ký tự user mỗi hội thoại, còn stress có 9261 ký tự user trong 16 lượt.
Lịch sử stress có khả năng vượt ngưỡng nhiều lần khi tính thêm phản hồi.
Đây chưa phải kết quả benchmark; cần kiểm chứng số compaction sau khi hoàn thiện
memory và agent. Test memory có thể dùng ngưỡng nhỏ hơn qua config riêng.

`load_config()` không khởi tạo model, không gọi API và không yêu cầu API key.
`build_chat_model()` chỉ import SDK của provider được chọn khi dựng model live.
Cloud provider thiếu key trả về `None`, để agent có thể dùng nhánh tất định.
Custom không có key dùng placeholder `not-required` theo yêu cầu constructor
OpenAI SDK; Ollama không nhận API key. Chỉ gọi hàm dựng model trong nhánh live
khi triển khai các CP agent sau này.

`normalize_provider()` bỏ khoảng trắng, chuyển về chữ thường và sửa alias
`anthorpic` thành `anthropic`. Provider không hỗ trợ bị từ chối bằng `ValueError`
kèm danh sách tên hợp lệ. Sai ngưỡng compact hoặc temperature cũng báo lỗi
ngay tại bước nạp config, kèm tên biến môi trường cần sửa.

### Kiểm tra CP3

Chạy từ repo root:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from config import load_config; cfg = load_config(); print(cfg.data_dir, '|', cfg.state_dir, '|', cfg.compact_threshold_tokens)"
```

Test config/provider, bao gồm khởi tạo model bằng key giả mà không gửi request:

```bash
.venv/Scripts/python.exe -m pytest src/test_config_provider.py -q
```

Các test SDK được skip nếu chưa cài package tương ứng. CP4 hoàn thiện hai test
lưu/sửa `User.md` và compact trigger trong `src/test_agents.py`; hai test hành vi
agent còn lại vẫn là scaffold dành cho các CP tiếp theo.

## CP4: memory layer

### Ước lượng token

`estimate_tokens()` dùng công thức của gợi ý Codelab: bỏ khoảng trắng ở hai đầu,
trả `0` cho chuỗi rỗng hoặc `None`, còn lại dùng `max(1, len(stripped) // 4)`.
Ví dụ: `a` và `abcde` đều là 1 token ước lượng; `abcdefgh` là 2. Đây là phép đo
heuristic theo ký tự, không phải tokenizer của provider. Cùng đầu vào luôn cho
cùng kết quả; không phụ thuộc model, thời gian hay random.

### Persistent profile

Khởi tạo `UserProfileStore(config.state_dir / "profiles")`. Với ID thông thường
như `dungct`, file nằm tại `state/profiles/dungct/User.md`. ID được trim khoảng
trắng; ID rỗng bị từ chối. ID lowercase ASCII bắt đầu bằng chữ/số, dài tối đa
64 ký tự và không phải tên dành riêng của Windows được dùng nguyên vẹn.
Các ID khác dùng thư mục `__user_<SHA256 đầy đủ>`. Namespace sinh ra được dành
riêng, nên nhập lại tên thư mục đó như một user ID không truy cập profile gốc.
Đường dẫn resolved cũng phải nằm dưới profile root, kể cả khi có symlink.

`read_text()` trả chuỗi rỗng nếu chưa có file; `write_text()` tạo thư mục cha và
ghi UTF-8; `edit_text()` thay đúng một lần và trả `False` khi không thay đổi.
`file_size()` lấy kích thước byte thực trên đĩa, trả `0` nếu file chưa tồn tại.
Việc ghi giữ nguyên newline để số byte UTF-8 lặp lại được trên Windows.

Facts dùng format `- key: value`. `facts()` đọc thành dict; `upsert_fact()` cập
nhật cùng khóa thay vì nối thêm giá trị cũ, dọn duplicate của khóa đó và giữ lại
ghi chú thủ công. Ghi lại cùng fact không làm thay đổi file hoặc tăng kích thước.

### Trích fact và correction

Các khóa chính là `name`, `profession`, `location`, `response_style`; bổ sung
`favorite_drink`, `favorite_food`, `pet`, `interests` khi có câu xác nhận phù hợp.
Đây là bộ quy tắc tiếng Việt tất định cho bài lab, không phải NLP tổng quát.

Correction được trích thành giá trị mới: Đà Nẵng → Huế và backend engineer →
MLOps engineer. Khi replay dữ liệu stress, correction nơi ở cuối cùng là
Đà Nẵng. Caller ghi dict cập nhật bằng `upsert_fact()` để profile không còn giữ
hai giá trị xung đột cho cùng khóa.

Bỏ các câu có dấu hỏi, câu đùa/giả định, phủ định không cung cấp giá trị mới và
thông tin đi họp/công tác/du lịch ngắn hạn. Một câu khai báo đứng trước câu hỏi
riêng vẫn được giữ. Chỉ nhắc tên trường, ví dụ "đồ uống yêu thích và style trả
lời", không phải cung cấp giá trị; nhãn món ăn/đồ uống cần `là` hoặc `:`.

### Compact theo thread

`append()` tính token ước lượng của summary cộng các message đang giữ. Khi vượt
ngưỡng và có message cũ để loại khỏi lịch sử nguyên văn, nó tạo summary và giữ
đúng `keep_messages` message gần nhất. Mỗi lần nén tăng counter một lần.
`context()` trả snapshot gồm `messages`, `summary`, `compactions`, tránh caller
sửa nhầm state. Thread khác nhau không chia sẻ lịch sử hay counter.

Summary dùng fact có cấu trúc và các đoạn context ngắn. Fact được cache đầy đủ
trong state nội bộ; summary chỉ nhận nguyên dòng fact nếu vừa ngân sách.
Fact quá dài không bị cắt thành giá trị sai và không ngăn fact ngắn phía sau
được đưa vào summary. Fact bị lược khỏi summary vẫn có thể được đưa lại khi
ngân sách tăng. Cache trích fact dùng các khóa cố định của extractor.

Phần summary có giới hạn `4 * max(1, threshold_tokens // 3)` ký tự. Phần context
có thể mất chi tiết; persistent `User.md` vẫn là nguồn fact qua nhiều thread.
Nếu riêng các message gần nhất đã quá dài, chúng vẫn được giữ nguyên văn;
threshold là điều kiện kích hoạt, không phải bảo đảm prompt luôn dưới ngưỡng.
Không tăng counter khi chưa có message cũ để nén. `keep_messages=0` cũng được
manager xử lý đúng, dù config CP3 chỉ nhận giá trị keep dương.

### Kiểm chứng CP4

```bash
.venv/Scripts/python.exe -m pytest src/test_memory_store.py src/test_agents.py::test_user_markdown_read_write_edit src/test_agents.py::test_compact_trigger -q
```

Kết quả: **56 passed**. Test bao gồm UTF-8, kích thước byte, persistence, edit
một lần, idempotence, đường dẫn/user ID, correction, nhiễu, summary nguyên fact,
khôi phục fact khi tăng budget, tail nguyên văn, thread isolation và replay hai
dataset cố định. Review phát hiện hai lỗi ID namespace và cắt dở fact; cả hai
được tái hiện bằng regression test, sửa và review lại.

Chạy toàn bộ `pytest src -q`: **96 passed, 2 failed**. Hai test còn thất bại tại
TODO vốn có là `test_cross_session_recall` và
`test_compact_reduces_prompt_load_on_long_thread`; chúng cần agent ở CP sau.

Replay ở mức memory layer, dùng default 1000/6 và phản hồi cố định
`Đã ghi nhận thông tin.` sau mỗi lượt user:

| Dataset | Threads | Compactions | Prompt ước lượng không compact | Prompt ước lượng có compact |
|---|---:|---:|---:|---:|
| `conversations.json` | 10 | 0 | 10668 | 10668 |
| `advanced_long_context.json` | 1 | 5 | 21073 | 11735 |

Các cột prompt là tổng context ước lượng sau từng lượt, không phải token API.
Replay này chưa dùng `reply()` của agent và chưa cộng `User.md` vào prompt;
không thay thế hai bảng benchmark cần hoàn thiện sau. Hai dataset không sửa.
