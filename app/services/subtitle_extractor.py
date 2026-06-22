"""Extract Chinese subtitles from video files or downloaded subtitle files.

Tries subtitle files first (downloaded by yt-dlp), then checks embedded
subtitle streams inside the MP4 container. If neither is available, returns
None so the caller can fall back to Whisper speech recognition.
"""

import json
import logging
import re
import subprocess
from pathlib import Path

from app.models.pipeline import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

# Subtitle file extensions we can parse, in order of preference
_SUB_EXTENSIONS = [".srt", ".vtt", ".ass", ".ssa"]

# Language tags that indicate Chinese
_CHINESE_LANGS = {"zh", "zh-hans", "zh-cn", "zh-tw", "zh-hant", "chi", "zho"}


class SubtitleExtractor:
    """Extracts Chinese subtitles and converts them to TranscriptionResult.

    Priority order:
    1. Subtitle files in work_dir (downloaded by yt-dlp alongside the video)
    2. Embedded subtitle streams inside the MP4 container
    3. Returns None → caller falls back to Whisper
    """

    def try_extract(self, video_path: Path, work_dir: Path) -> "TranscriptionResult | None":
        """Attempt to extract subtitles. Returns None if none found."""
        result = self._from_subtitle_files(work_dir)
        if result:
            logger.info("Using subtitle file from work_dir for %s", video_path.name)
            return result

        result = self._from_embedded_streams(video_path, work_dir)
        if result:
            logger.info("Using embedded subtitle stream from %s", video_path.name)
            return result

        logger.info("No subtitles found for %s — will use Whisper", video_path.name)
        return None

    # ── File-based subtitles ─────────────────────────────────────────────────

    def _from_subtitle_files(self, work_dir: Path) -> "TranscriptionResult | None":
        """Find subtitle files downloaded by yt-dlp (e.g., original.zh.srt)."""
        candidates: list[Path] = []
        for ext in _SUB_EXTENSIONS:
            candidates.extend(work_dir.glob(f"original*{ext}"))

        if not candidates:
            return None

        # Prefer Chinese-tagged files; fall back to any
        zh_files = [
            p for p in candidates
            if any(lang in p.stem.lower().replace("-", "").replace("_", "") for lang in _CHINESE_LANGS)
        ]
        target = zh_files[0] if zh_files else candidates[0]
        logger.debug("Parsing subtitle file: %s", target)
        return self._parse_file(target)

    # ── Embedded subtitle streams ────────────────────────────────────────────

    def _from_embedded_streams(
        self, video_path: Path, work_dir: Path
    ) -> "TranscriptionResult | None":
        """Extract an embedded subtitle stream using ffprobe + ffmpeg."""
        streams = self._list_subtitle_streams(video_path)
        if not streams:
            return None

        # Prefer Chinese-tagged stream; otherwise take the first one
        target = None
        for s in streams:
            lang = s.get("tags", {}).get("language", "").lower()
            if lang in _CHINESE_LANGS or not lang:  # untagged streams are often Chinese on Douyin
                target = s
                break
        if target is None:
            target = streams[0]

        stream_idx = target.get("index", 0)
        srt_path = work_dir / "embedded_subs.srt"

        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", str(video_path),
            "-map", f"0:{stream_idx}",
            str(srt_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
        except Exception as exc:
            logger.warning("ffmpeg subtitle extraction failed: %s", exc)
            return None

        if result.returncode != 0 or not srt_path.exists():
            return None

        return self._parse_srt(srt_path)

    def _list_subtitle_streams(self, video_path: Path) -> list[dict]:
        """Use ffprobe to list subtitle streams in the video container."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "s",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return []
            data = json.loads(result.stdout)
            return data.get("streams", [])
        except Exception as exc:
            logger.debug("ffprobe subtitle probe failed: %s", exc)
            return []

    # ── Parsers ──────────────────────────────────────────────────────────────

    def _parse_file(self, path: Path) -> "TranscriptionResult | None":
        suffix = path.suffix.lower()
        if suffix == ".srt":
            return self._parse_srt(path)
        if suffix == ".vtt":
            return self._parse_vtt(path)
        if suffix in (".ass", ".ssa"):
            return self._parse_ass(path)
        return None

    def _parse_srt(self, path: Path) -> "TranscriptionResult | None":
        content = _read_text(path)
        if not content:
            return None

        segments: list[TranscriptionSegment] = []
        # Split on blank lines between cues
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue
            # Skip index line if numeric
            start_line = 1 if re.match(r"^\d+$", lines[0].strip()) else 0
            timing = _parse_srt_timing(lines[start_line])
            if timing is None:
                continue
            start, end = timing
            text = _clean_html(" ".join(lines[start_line + 1:]))
            if text:
                segments.append(
                    TranscriptionSegment(start=start, end=end, text=text, speaker="speaker_1")
                )

        return _make_result(segments)

    def _parse_vtt(self, path: Path) -> "TranscriptionResult | None":
        content = _read_text(path)
        if not content:
            return None

        segments: list[TranscriptionSegment] = []
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = [ln for ln in block.strip().splitlines() if not ln.strip().startswith("NOTE")]
            if not lines:
                continue
            # VTT timing: 00:00:01.000 --> 00:00:02.500
            timing_line = next((ln for ln in lines if "-->" in ln), None)
            if not timing_line:
                continue
            timing = _parse_vtt_timing(timing_line)
            if timing is None:
                continue
            start, end = timing
            text_lines = lines[lines.index(timing_line) + 1:]
            text = _clean_html(" ".join(text_lines))
            if text:
                segments.append(
                    TranscriptionSegment(start=start, end=end, text=text, speaker="speaker_1")
                )

        return _make_result(segments)

    def _parse_ass(self, path: Path) -> "TranscriptionResult | None":
        content = _read_text(path)
        if not content:
            return None

        segments: list[TranscriptionSegment] = []
        # ASS Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        pattern = re.compile(
            r"^Dialogue:\s*\d+,"
            r"(\d+):(\d{2}):(\d{2})\.(\d{2}),"
            r"(\d+):(\d{2}):(\d{2})\.(\d{2}),"
            r"[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,"
            r"(.*)$",
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            start = (
                int(m.group(1)) * 3600
                + int(m.group(2)) * 60
                + int(m.group(3))
                + int(m.group(4)) / 100
            )
            end = (
                int(m.group(5)) * 3600
                + int(m.group(6)) * 60
                + int(m.group(7))
                + int(m.group(8)) / 100
            )
            text = re.sub(r"\{[^}]*\}", "", m.group(9))  # remove ASS tags
            text = text.replace("\\N", " ").replace("\\n", " ").strip()
            if text:
                segments.append(
                    TranscriptionSegment(start=start, end=end, text=text, speaker="speaker_1")
                )

        segments.sort(key=lambda s: s.start)
        return _make_result(segments)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_text(path: Path) -> str:
    """Read subtitle file, trying UTF-8-BOM then UTF-8 then GBK."""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb2312"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_srt_timing(line: str) -> "tuple[float, float] | None":
    m = re.match(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
        line.strip(),
    )
    if not m:
        return None
    start = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
    end = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
    return start, end


def _parse_vtt_timing(line: str) -> "tuple[float, float] | None":
    # VTT: 00:00:01.000 --> 00:00:02.500 (may have hours or not)
    m = re.match(
        r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})",
        line.strip(),
    )
    if not m:
        return None
    h1 = int(m[1] or 0)
    h2 = int(m[5] or 0)
    start = h1 * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
    end = h2 * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
    return start, end


def _clean_html(text: str) -> str:
    """Remove HTML/XML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(text.split())


def _make_result(segments: list[TranscriptionSegment]) -> "TranscriptionResult | None":
    if not segments:
        return None
    full_text = " ".join(s.text for s in segments)
    return TranscriptionResult(
        segments=segments,
        full_text=full_text,
        language="zh",
        confidence=1.0,
    )
