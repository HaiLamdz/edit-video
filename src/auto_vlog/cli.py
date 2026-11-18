"""Command-line interface for the AI Daily Vlog Auto-Editor."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from auto_vlog.config import Settings
from auto_vlog.director import run_director
from auto_vlog.ingest import VIDEO_EXTENSIONS, run_ingest
from auto_vlog.render import find_background_music, run_render
from auto_vlog.voice_sub import run_voice_and_subtitles


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def _cmd_doctor(settings: Settings) -> int:
    settings.ensure_dirs()
    ffmpeg_ok = bool(shutil.which("ffmpeg"))
    ffprobe_ok = bool(shutil.which("ffprobe"))
    key_ok = bool(settings.openai_api_key)
    clips = [
        p for p in settings.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    music = find_background_music(settings.music_dir)

    print("=== Kiểm tra môi trường auto-vlog ===")
    print(f"- ffmpeg        : {'OK' if ffmpeg_ok else 'THIẾU (cần cài ffmpeg)'}")
    print(f"- ffprobe       : {'OK' if ffprobe_ok else 'THIẾU (đi kèm ffmpeg)'}")
    print(f"- OPENAI_API_KEY: {'OK' if key_ok else 'CHƯA CẤU HÌNH (kiểm tra .env)'}")
    print(f"- TTS provider  : {settings.tts_provider}")
    print(f"- input_clips/  : {len(clips)} video")
    print(f"- music/        : {music.name if music else 'Chưa có nhạc nền (không bắt buộc)'}")
    return 0 if (ffmpeg_ok and ffprobe_ok and key_ok) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-vlog",
        description="Tự động dựng vlog hằng ngày 40s-60s bằng OpenAI API + FFmpeg.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Hiện log chi tiết")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Kiểm tra ffmpeg, API key và thư mục đầu vào")
    sub.add_parser("ingest", help="Bước 1: Đọc metadata, xếp theo giờ quay và trích khung hình")

    p_dir = sub.add_parser("director", help="Bước 2: AI xem clip và viết kịch bản JSON")
    p_dir.add_argument("--note", default="", help="Ghi chú bối cảnh (VD: '1 ngày đi Tam Đảo')")
    p_dir.add_argument("--refresh", action="store_true", help="Bỏ cache Vision và phân tích lại")

    sub.add_parser("voice", help="Bước 3: Tạo voiceover từng nhịp, căn khớp hình và sinh phụ đề .ass")

    p_ren = sub.add_parser("render", help="Bước 4: Cắt ghép mượt, trộn nhạc ducking và xuất video")
    p_ren.add_argument("--output", default="vlog_final.mp4", help="Tên file kết quả trong output/")

    p_run = sub.add_parser("run", help="Chạy toàn bộ pipeline (Bước 1 -> 4)")
    p_run.add_argument("--note", default="", help="Ghi chú bối cảnh (VD: '1 ngày đi Tam Đảo')")
    p_run.add_argument("--refresh", action="store_true", help="Bỏ cache Vision và phân tích lại")
    p_run.add_argument("--output", default="vlog_final.mp4", help="Tên file kết quả trong output/")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    settings = Settings.from_env(Path.cwd())

    try:
        if args.command == "doctor":
            return _cmd_doctor(settings)

        if args.command == "ingest":
            manifest = run_ingest(settings)
            print(f"Đã ingest {len(manifest.clips)} clip -> {settings.work_dir / 'ingest_manifest.json'}")
            return 0

        if args.command == "director":
            plan = run_director(settings, user_note=args.note, refresh_cache=args.refresh)
            print(f"Đã tạo kịch bản '{plan.title}' ({len(plan.beats)} nhịp) -> {settings.work_dir / 'vlog_plan.json'}")
            print("Mẹo: Bạn có thể mở `work/vlog_plan.json` để sửa lời bình trước khi chạy bước `voice`.")
            return 0

        if args.command == "voice":
            timeline = run_voice_and_subtitles(settings)
            print(f"Đã tạo voice + phụ đề ({timeline.total_duration_sec:.1f}s) -> {settings.work_dir / 'timeline.json'}")
            return 0

        if args.command == "render":
            out = run_render(settings, output_filename=args.output)
            print(f"Xong! Video của bạn đã lưu tại: {out}")
            return 0

        if args.command == "run":
            manifest = run_ingest(settings)
            print(f"[1/4] Ingest xong {len(manifest.clips)} clip.")
            plan = run_director(settings, user_note=args.note, refresh_cache=args.refresh)
            print(f"[2/4] Đạo diễn AI xong kịch bản '{plan.title}' ({len(plan.beats)} nhịp).")
            timeline = run_voice_and_subtitles(settings)
            print(f"[3/4] Tạo voiceover & phụ đề xong (tổng {timeline.total_duration_sec:.1f}s).")
            out = run_render(settings, output_filename=args.output)
            print(f"[4/4] Render hoàn tất -> {out}")
            return 0

    except Exception as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
