"""Speaker gender detection using pitch analysis on isolated vocals audio."""

import logging
import struct
import subprocess
from collections import defaultdict
from pathlib import Path

from app.models.pipeline import TranscriptionResult

logger = logging.getLogger(__name__)

FEMALE_PITCH_HZ_THRESHOLD = 165  # Hz: above → female, below/equal → male
SAMPLE_RATE = 8000


class GenderDetector:
    """Estimates speaker gender (male/female) via fundamental frequency analysis.

    Uses ffmpeg to extract short raw PCM clips for each speaker, then
    estimates pitch via autocorrelation. Median over 100ms windows reduces
    noise from consonants and pauses.
    """

    def detect(
        self, vocals_path: Path, transcription: TranscriptionResult
    ) -> dict[str, str]:
        """Return {speaker_id: 'male'|'female'|'unknown'} for each speaker."""
        speaker_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for seg in transcription.segments:
            speaker = seg.speaker or "speaker_1"
            speaker_intervals[speaker].append((seg.start, seg.end))

        gender_map: dict[str, str] = {}
        for speaker, intervals in speaker_intervals.items():
            best_start, best_end = max(intervals, key=lambda iv: iv[1] - iv[0])
            duration = best_end - best_start
            if duration < 0.5:
                gender_map[speaker] = "unknown"
                continue

            pitch = self._estimate_pitch(vocals_path, best_start, min(duration, 4.0))
            if pitch <= 0:
                gender_map[speaker] = "unknown"
            elif pitch >= FEMALE_PITCH_HZ_THRESHOLD:
                gender_map[speaker] = "female"
            else:
                gender_map[speaker] = "male"

            logger.info(
                "Speaker %s: estimated pitch %.1f Hz → %s",
                speaker, pitch, gender_map[speaker],
            )

        return gender_map

    def _estimate_pitch(self, audio_path: Path, start: float, duration: float) -> float:
        """Estimate fundamental frequency using autocorrelation on PCM samples.

        Analyzes up to 5 evenly-spaced 100ms windows and returns the median pitch.
        """
        raw = self._extract_pcm(audio_path, start, duration)
        if not raw or len(raw) < 400:
            return 0.0

        n = len(raw) // 2
        all_samples = list(struct.unpack(f"{n}h", raw[:n * 2]))

        window_len = SAMPLE_RATE // 10  # 100ms
        if n < window_len:
            return 0.0

        # Pick 5 evenly-spaced windows to average across the segment
        step = max(1, (n - window_len) // 4)
        pitches = []
        for k in range(5):
            offset = k * step
            if offset + window_len > n:
                break
            window = all_samples[offset : offset + window_len]
            p = self._autocorr_pitch(window)
            if p > 0:
                pitches.append(p)

        if not pitches:
            return 0.0

        pitches.sort()
        return pitches[len(pitches) // 2]

    def _autocorr_pitch(self, samples: list[int]) -> float:
        """Autocorrelation pitch detector for a single 100ms window."""
        mean = sum(samples) / len(samples)
        s = [x - mean for x in samples]

        energy = sum(x * x for x in s)
        if energy < 1_000_000:  # silence or very quiet — skip
            return 0.0

        n = len(s)
        min_lag = SAMPLE_RATE // 400  # 400 Hz upper bound (highest female)
        max_lag = SAMPLE_RATE // 50   # 50 Hz lower bound (lowest male)

        best_corr = -1.0
        best_lag = min_lag

        for lag in range(min_lag, min(max_lag + 1, n // 2)):
            corr = sum(s[i] * s[i + lag] for i in range(n - lag)) / energy
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        if best_corr < 0.25:  # low periodicity → unvoiced segment
            return 0.0

        return SAMPLE_RATE / best_lag

    def _extract_pcm(self, audio_path: Path, start: float, duration: float) -> bytes:
        """Extract raw 16-bit signed mono PCM via ffmpeg."""
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-ss", str(start), "-t", str(duration),
            "-i", str(audio_path),
            "-filter:a",
            f"aresample={SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=mono",
            "-f", "s16le", "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=20)
            if result.returncode == 0:
                return result.stdout
        except Exception as exc:
            logger.warning("PCM extraction failed for %s: %s", audio_path, exc)
        return b""
