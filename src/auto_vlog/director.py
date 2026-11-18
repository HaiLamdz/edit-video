"""Stage 2: Two-pass OpenAI Vision clip analysis and storytelling script generation."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from openai import OpenAI

from auto_vlog.config import Settings
from auto_vlog.models import (
    ClipAnalysis,
    ClipMetadata,
    IngestManifest,
    SelectedSegment,
    StoryBeat,
    VlogPlan,
)

logger = logging.getLogger(__name__)


def _build_openai_client(settings: Settings) -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Hãy copy `.env.example` thành `.env` và điền key OpenAI của bạn."
        )
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def _encode_image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _parse_structured_completion(
    client: OpenAI,
    model: str,
    messages: list[dict],
    response_format: type,
):
    """Call OpenAI Structured Outputs using stable `.chat.completions.parse` with beta fallback."""
    parser_fn = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    return parser_fn(
        model=model,
        messages=messages,
        response_format=response_format,
    )


def analyze_single_clip(client: OpenAI, model: str, clip: ClipMetadata) -> ClipAnalysis:
    """Ask OpenAI Vision to summarize a clip and identify clean usable windows."""
    frame_timestamps = [f.timestamp_sec for f in clip.sampled_frames]
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Đây là các khung hình lấy mẫu theo thứ tự thời gian từ clip '{clip.clip_id}' "
                f"(tên file: {clip.filename}, tổng thời lượng: {clip.duration_sec:.2f}s, "
                f"giờ quay: {clip.creation_time}).\n"
                f"Các mốc giây của từng ảnh lần lượt là: {frame_timestamps}.\n"
                "Hãy mô tả ngắn gọn nội dung bằng tiếng Việt và đề xuất 1-2 khoảng giây đẹp, "
                "ít rung (best_windows) nằm trong [0, duration_sec] để dùng cắt dựng vlog. "
                "Đánh dấu keep_ambient_audio=true nếu cảnh phù hợp giữ tiếng môi trường nhẹ "
                "(đồ ăn xèo xèo, thiên nhiên, quán cà phê), false nếu khả năng ồn tiếng xe/gió."
            ),
        }
    ]

    for frame in clip.sampled_frames:
        img_path = Path(frame.image_path)
        if not img_path.exists():
            continue
        b64 = _encode_image_b64(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                },
            }
        )

    completion = _parse_structured_completion(
        client=client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý dựng phim vlog chuyên nghiệp. "
                    "Hãy phân tích khách quan nội dung hình ảnh và trả về đúng schema."
                ),
            },
            {"role": "user", "content": content},
        ],
        response_format=ClipAnalysis,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"OpenAI không trả về dữ liệu hợp lệ cho {clip.clip_id}.")
    parsed.clip_id = clip.clip_id
    return parsed


def _load_cached_analyses(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def analyze_all_clips(
    client: OpenAI,
    settings: Settings,
    manifest: IngestManifest,
    refresh_cache: bool = False,
) -> list[ClipAnalysis]:
    """Analyze every clip in the manifest with disk caching to avoid duplicate Vision billing."""
    cache_path = settings.work_dir / "clip_analyses.json"
    cache = {} if refresh_cache else _load_cached_analyses(cache_path)

    results: list[ClipAnalysis] = []
    for clip in manifest.clips:
        cache_key = f"{clip.clip_id}:{clip.filename}:{clip.duration_sec:.2f}"
        if cache_key in cache:
            results.append(ClipAnalysis.model_validate(cache[cache_key]))
            continue

        logger.info("Analyzing %s with OpenAI Vision (%s)...", clip.clip_id, settings.openai_vision_model)
        analysis = analyze_single_clip(client, settings.openai_vision_model, clip)
        cache[cache_key] = analysis.model_dump()
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(analysis)

    return results


def sanitize_vlog_plan(plan: VlogPlan, manifest: IngestManifest) -> VlogPlan:
    """Clamp segment timestamps strictly to actual clip durations and drop unknown clip IDs."""
    clips_by_id = {c.clip_id: c for c in manifest.clips}
    clean_beats: list[StoryBeat] = []

    for idx, beat in enumerate(plan.beats, start=1):
        valid_segments: list[SelectedSegment] = []
        for seg in beat.segments:
            clip = clips_by_id.get(seg.clip_id)
            if not clip:
                continue
            start = max(0.0, min(seg.start_sec, max(0.0, clip.duration_sec - 0.5)))
            end = min(clip.duration_sec, max(start + 0.5, seg.end_sec))
            if end - start < 0.4:
                continue
            valid_segments.append(
                SelectedSegment(
                    clip_id=seg.clip_id,
                    start_sec=round(start, 2),
                    end_sec=round(end, 2),
                    keep_ambient_audio=seg.keep_ambient_audio,
                    visual_note=seg.visual_note,
                )
            )

        if valid_segments:
            clean_beats.append(
                StoryBeat(
                    beat_index=len(clean_beats) + 1,
                    time_label=beat.time_label.strip(),
                    narration=beat.narration.strip(),
                    transition_to_next=beat.transition_to_next,
                    segments=valid_segments,
                )
            )
        else:
            logger.warning("Dropped beat %d because it had no valid clip references.", idx)

    if not clean_beats:
        raise RuntimeError("Kịch bản AI trả về không có đoạn clip hợp lệ nào.")

    return VlogPlan(title=plan.title.strip(), mood=plan.mood.strip(), beats=clean_beats)


def build_vlog_plan(
    client: OpenAI,
    settings: Settings,
    manifest: IngestManifest,
    analyses: list[ClipAnalysis],
    user_note: str = "",
) -> VlogPlan:
    """Synthesize a 40-60s chronological storytelling plan from clip analyses."""
    clips_summary = []
    analysis_map = {a.clip_id: a for a in analyses}
    for clip in manifest.clips:
        a = analysis_map.get(clip.clip_id)
        clips_summary.append(
            {
                "clip_id": clip.clip_id,
                "creation_time": clip.creation_time,
                "duration_sec": clip.duration_sec,
                "summary": a.summary if a else "",
                "best_windows": [w.model_dump() for w in (a.best_windows if a else [])],
            }
        )

    target_mid = round((settings.target_min_sec + settings.target_max_sec) / 2)
    min_words = int(settings.target_min_sec * 2.4)
    max_words = int(settings.target_max_sec * 2.8)

    prompt = (
        f"Bạn là đạo diễn dựng vlog ngắn phong cách đời thường (giống vlog '1 ngày đi Tam Đảo', '1 ngày đi làm').\n"
        f"Ghi chú thêm của người quay: {user_note or 'Không có ghi chú thêm, hãy dựa theo hình ảnh và giờ quay.'}\n\n"
        f"Mục tiêu video:\n"
        f"- Tổng thời lượng mong muốn: {settings.target_min_sec}-{settings.target_max_sec} giây (lý tưởng ~{target_mid}s).\n"
        f"- Chia thành 5 đến 7 nhịp truyện (beats) theo ĐÚNG trình tự thời gian từ sáng đến tối.\n"
        f"- Tổng độ dài lời bình (narration) tất cả các beat cộng lại khoảng {min_words}-{max_words} từ tiếng Việt.\n"
        f"- Giọng văn kể chuyện tự nhiên, gần gũi ngôi thứ nhất (không sáo rỗng, không dùng từ quảng cáo).\n"
        f"- Mỗi beat chọn 1-3 đoạn clip minh họa khớp nội dung lời bình; mỗi đoạn dài khoảng 1.5-3.5s.\n"
        f"- Dùng chính xác `clip_id` và chọn `start_sec`, `end_sec` nằm trong `duration_sec` của clip đó.\n"
        f"- `time_label`: ngắn gọn kiểu '07:30 • Xuất phát' hoặc để trống nếu không rõ.\n"
        f"- `transition_to_next`: ưu tiên 'fade' hoặc 'dissolve' khi đổi địa điểm/buổi; dùng 'none' nếu nối nhanh cùng bối cảnh.\n\n"
        f"Danh sách clip theo thứ tự thời gian:\n{json.dumps(clips_summary, ensure_ascii=False, indent=2)}"
    )

    completion = _parse_structured_completion(
        client=client,
        model=settings.openai_director_model,
        messages=[
            {
                "role": "system",
                "content": "Bạn là đạo diễn vlog TikTok/Reels tiếng Việt. Luôn trả về JSON đúng schema.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=VlogPlan,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("OpenAI Director không trả về kịch bản hợp lệ.")
    return sanitize_vlog_plan(parsed, manifest)


def run_director(
    settings: Settings,
    user_note: str = "",
    refresh_cache: bool = False,
) -> VlogPlan:
    """Execute Stage 2 using `work/ingest_manifest.json` and save `work/vlog_plan.json`."""
    manifest_path = settings.work_dir / "ingest_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Chưa có `work/ingest_manifest.json`. Hãy chạy bước `ingest` trước.")

    manifest = IngestManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    client = _build_openai_client(settings)

    analyses = analyze_all_clips(client, settings, manifest, refresh_cache=refresh_cache)
    plan = build_vlog_plan(client, settings, manifest, analyses, user_note=user_note)

    plan_path = settings.work_dir / "vlog_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return plan
