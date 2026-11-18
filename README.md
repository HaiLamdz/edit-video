# AI Daily Vlog Auto-Editor (`auto-vlog`)

Khung dự án Python tự động dựng vlog ngắn **40–60 giây** theo phong cách kể chuyện (**Hướng B: Voiceover + Phụ đề + Chuyển cảnh mượt**), sử dụng **OpenAI API** và **FFmpeg**.

---

## Cấu trúc dự án

```text
ai-daily-vlog/
├── input_clips/          # Bỏ 10-30 clip quay thô trong ngày (.mp4/.mov) vào đây
├── music/                # (Tùy chọn) Bỏ 1 bài nhạc nền (.mp3/.wav) vào đây
├── output/               # Video thành phẩm (vlog_final.mp4)
├── work/                 # File trung gian (frames, vlog_plan.json, audio, subtitles.ass)
├── src/auto_vlog/
│   ├── config.py         # Đọc cấu hình từ .env
│   ├── models.py         # Pydantic schema giữa các bước
│   ├── media_utils.py    # Wrapper cho ffmpeg / ffprobe
│   ├── ingest.py         # Bước 1: Đọc metadata giờ quay, xếp thứ tự, trích khung hình
│   ├── director.py       # Bước 2: OpenAI Vision phân tích clip + viết kịch bản JSON
│   ├── voice_sub.py      # Bước 3: Tạo voiceover từng nhịp, căn khớp hình, tạo phụ đề .ass
│   ├── render.py         # Bước 4: FFmpeg ghép 9:16, xfade mượt, ducking nhạc, gắn phụ đề
│   └── cli.py            # Giao diện dòng lệnh
├── tests/                # Unit test logic timeline & phụ đề
└── main.py
```

---

## Cài đặt nhanh

### 1. Cài FFmpeg trên máy
- **Ubuntu / Debian:** `sudo apt update && sudo apt install -y ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Cài qua `winget install Gyan.FFmpeg` (hoặc tải bản static rồi thêm vào `PATH`)

### 2. Cài thư viện Python (khuyên dùng `uv`)
```bash
uv sync --extra dev
```
*(Hoặc dùng `pip`: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`)*

### 3. Cấu hình OpenAI API Key
```bash
cp .env.example .env
```
Mở `.env` và điền `OPENAI_API_KEY=sk-...` của bạn.

> **Mẹo giọng đọc tiếng Việt:** Mặc định `TTS_PROVIDER=openai` dùng luôn OpenAI API Key (`gpt-4o-mini-tts`). Nếu muốn thử giọng đọc tiếng Việt bản địa miễn phí để so sánh, đổi `TTS_PROVIDER=edge` trong `.env`.

---

## Cách sử dụng

1. Chép các clip quay trong ngày vào thư mục `input_clips/`.
2. *(Tùy chọn)* Chép 1 file nhạc nền nhẹ vào thư mục `music/`.
3. Kiểm tra môi trường:
   ```bash
   uv run python main.py doctor
   ```
4. Chạy toàn bộ pipeline:
   ```bash
   uv run python main.py run --note "1 ngày đi Tam Đảo với bạn, ăn gà nướng và uống cà phê sương mù"
   ```
5. Video hoàn chỉnh sẽ xuất tại `output/vlog_final.mp4`.

### Chạy từng bước (nếu muốn tự sửa kịch bản trước khi render)
```bash
uv run python main.py ingest
uv run python main.py director --note "1 ngày đi làm bình thường"
# -> Lúc này bạn có thể mở file work/vlog_plan.json để chỉnh lại câu chữ hoặc đổi đoạn cắt!
uv run python main.py voice
uv run python main.py render
```
