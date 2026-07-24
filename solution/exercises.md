# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> **Ở temperature 0.0, phản hồi ổn định, trực tiếp và ít thay đổi giữa các lần gọi. Khi tăng lên 0.5 và 1.0, cách diễn đạt và sự thật được chọn đa dạng hơn; ở 1.5, câu trả lời sáng tạo và khó đoán hơn nhưng cũng dễ lan man hoặc xuất hiện thông tin kém chính xác (ảo giác). Temperature càng cao thì mức độ ngẫu nhiên càng lớn.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *temperature = 0.2 để ổn định câu trả lời, đúng yêu cầu ngữ cảnh doanh nghiệp*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Workload tạo ra khoảng 10.000 × 3 × 350 = 10,5 triệu output token mỗi ngày. Với mức giá output thường được dùng trong bài cho GPT-4o là 15 USD/triệu token và GPT-4o-mini là 0,60 USD/triệu token, GPT-4o đắt hơn khoảng 25 lần nếu chỉ xét token đầu ra: khoảng 157,50 USD/ngày so với 6,30 USD/ngày. GPT-4o xứng đáng với chi phí khi xử lý yêu cầu phức tạp, nhiều sắc thái hoặc có rủi ro cao, chẳng hạn phân tích một hợp đồng dài; GPT-4o-mini phù hợp với các tác vụ khối lượng lớn và đơn giản như phân loại yêu cầu hỗ trợ, tóm tắt ngắn.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Phản hồi theo persona giáo viên tiểu học thường ngắn hơn, dùng từ quen thuộc và có ví dụ trực quan, chẳng hạn ví blockchain như một cuốn sổ chung mà nhiều người cùng giữ bản sao. Phản hồi theo persona chuyên gia tài chính dài và chi tiết hơn, sử dụng các thuật ngữ như sổ cái phân tán, cơ chế đồng thuận, hàm băm, tài sản mã hóa và rủi ro thanh khoản. System prompt không thay đổi câu hỏi của người dùng nhưng định hướng đối tượng độc giả, độ sâu, giọng điệu, từ vựng và cách tổ chức câu trả lời. Vì vậy, cùng một model có thể thể hiện những hành vi rất khác nhau tùy theo vai trò được quy định.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Tôi sử dụng một đoạn văn tiếng Việt dài khoảng 100 từ. Công thức số từ / 0.75 ước lượng khoảng 133 token, trong khi count_tokens bằng tiktoken cho kết quả khoảng 200 token, tức kết quả thực tế cao hơn khoảng 50% so với ước lượng. Nguyên nhân là token không tương ứng trực tiếp với từ: nhiều từ tiếng Việt có dấu hoặc các tổ hợp ký tự không được lưu thành một token duy nhất nên có thể bị tách thành nhiều token. Trong tiếng Anh, nhiều từ thông dụng xuất hiện thường xuyên trong dữ liệu huấn luyện nên thường được biểu diễn bằng ít token hơn.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất khi model cần tạo câu trả lời dài hoặc có thời gian phản hồi đáng kể, chẳng hạn chatbot hội thoại, trợ lý viết nội dung hoặc công cụ giải thích tài liệu, vì người dùng có thể đọc phần đầu ngay thay vì nhìn màn hình chờ. Cách này làm giảm cảm giác chậm dù tổng thời gian xử lý có thể không thay đổi nhiều. Non-streaming phù hợp hơn khi phản hồi rất ngắn, khi ứng dụng cần toàn bộ kết quả trước khi kiểm tra hoặc phân tích, hoặc khi đầu ra phải được xác thực theo một cấu trúc hoàn chỉnh như JSON trước khi hiển thị.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, ví dụ 1, 2, 4 rồi 8 giây, nên giảm số request gửi tới API trong lúc hệ thống đang quá tải và cho máy chủ thời gian phục hồi. Với delay cố định, hàng nghìn client có thể cùng thất bại rồi cùng retry sau đúng một giây, tạo thành các đợt request đồng loạt và tiếp tục làm hệ thống quá tải; hiện tượng này thường được gọi là “thundering herd”. Kết hợp exponential backoff với một khoảng ngẫu nhiên nhỏ, hay jitter, còn giúp phân tán thời điểm retry của các client.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Tôi chọn persona trợ lý học lập trình Python dành cho người mới bắt đầu. System prompt của tôi là: Bạn là trợ lý học Python thân thiện dành cho người mới bắt đầu. Hãy trả lời bằng tiếng Việt, giải thích ngắn gọn theo từng bước và đưa ra ví dụ mã nguồn nhỏ, có thể chạy được. Không cung cấp ngay toàn bộ lời giải cho bài tập; trước tiên hãy đưa ra gợi ý để người học tự suy nghĩ. Khi phát hiện lỗi trong mã, hãy giải thích nguyên nhân trước khi đề xuất cách sửa. Nếu không chắc chắn, hãy nói rõ thay vì tự tạo thông tin.
Tôi yêu cầu giải thích ngắn gọn theo từng bước để người mới không bị quá tải bởi nhiều thuật ngữ cùng lúc. Yêu cầu không cung cấp ngay toàn bộ lời giải giúp trợ lý đóng vai trò hướng dẫn học tập thay vì chỉ làm bài hộ người dùng. Việc chỉ định tiếng Việt cũng giúp câu trả lời nhất quán và phù hợp với nhóm người dùng mục tiêu.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý hiện tại là chỉ lưu ba lượt hội thoại gần nhất, vì vậy nó có thể quên mục tiêu học tập, lỗi đã gặp hoặc kiến thức đã giải thích trước đó. Tôi sẽ cải thiện bằng cách lưu lịch sử hội thoại vào cơ sở dữ liệu và tạo một bản tóm tắt ngắn sau mỗi số lượt nhất định. Khi người dùng gửi câu hỏi mới, hệ thống sẽ đưa bản tóm tắt cùng các tin nhắn gần nhất vào context; nếu dữ liệu nhiều hơn, có thể dùng embedding và vector database để truy xuất những đoạn lịch sử liên quan. Cách này duy trì được ngữ cảnh dài hạn mà không phải gửi toàn bộ cuộc trò chuyện trong mỗi API call.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
