"""Vocal isolation service using Demucs for separating vocals from background audio."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from app.models.pipeline import VocalIsolationResult

logger = logging.getLogger(__name__)


class VocalIsolator:
    """Separates vocals from background music using Demucs (Meta Research).

    Uses Demucs with --two-stems vocals mode to produce two output files:
    - vocals.wav: isolated vocal track
    - no_vocals.wav: background music/instrumental

    The class is stateless and calls Demucs via subprocess for simplicity.
    """

    def isolate(self, audio_path: Path, output_dir: Path) -> VocalIsolationResult:
        """Separate vocals from background audio using Demucs.

        Args:
            audio_path: Path to the input WAV audio file.
            output_dir: Directory where separated stems will be saved.

        Returns:
            VocalIsolationResult with paths to vocals and background audio files.
            On failure, both paths point to a copy of the original audio (graceful degradation).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "demucs",
                    "--two-stems",
                    "vocals",
                    "-o",
                    str(output_dir),
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for long audio
            )

            if result.returncode != 0:
                logger.warning(
                    "Demucs process failed with return code %d: %s",
                    result.returncode,
                    result.stderr,
                )
                return self._fallback(audio_path, output_dir)

            # Demucs outputs to: <output_dir>/htdemucs/<track_name>/vocals.wav and no_vocals.wav
            track_name = audio_path.stem
            stems_dir = output_dir / "htdemucs" / track_name

            vocals_path = stems_dir / "vocals.wav"
            no_vocals_path = stems_dir / "no_vocals.wav"

            if not vocals_path.exists() or not no_vocals_path.exists():
                logger.warning(
                    "Demucs output files not found at expected paths: %s, %s",
                    vocals_path,
                    no_vocals_path,
                )
                return self._fallback(audio_path, output_dir)

            # Move files to output_dir root for cleaner access
            final_vocals = output_dir / "vocals.wav"
            final_background = output_dir / "background.wav"

            shutil.move(str(vocals_path), str(final_vocals))
            shutil.move(str(no_vocals_path), str(final_background))

            # Clean up the intermediate demucs directory structure
            htdemucs_dir = output_dir / "htdemucs"
            if htdemucs_dir.exists():
                shutil.rmtree(str(htdemucs_dir))

            logger.info("Vocal isolation completed successfully for %s", audio_path.name)
            return VocalIsolationResult(
                vocals_path=final_vocals,
                background_path=final_background,
            )

        except subprocess.TimeoutExpired:
            logger.warning(
                "Demucs timed out after 600s for %s, using fallback",
                audio_path.name,
            )
            return self._fallback(audio_path, output_dir)
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(
                "Demucs execution failed for %s: %s, using fallback",
                audio_path.name,
                str(e),
            )
            return self._fallback(audio_path, output_dir)

    def _fallback(self, audio_path: Path, output_dir: Path) -> VocalIsolationResult:
        """Graceful degradation: copy original audio as both vocals and background.

        This allows the pipeline to continue even when vocal isolation fails.
        The transcription quality may be lower due to background music interference.
        """
        logger.warning(
            "Using fallback for vocal isolation: copying original audio as vocals and background"
        )

        fallback_vocals = output_dir / "vocals.wav"
        fallback_background = output_dir / "background.wav"

        shutil.copy2(str(audio_path), str(fallback_vocals))
        shutil.copy2(str(audio_path), str(fallback_background))

        return VocalIsolationResult(
            vocals_path=fallback_vocals,
            background_path=fallback_background,
        )
