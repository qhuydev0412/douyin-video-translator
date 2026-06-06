"""Script chạy pipeline dịch video Douyin trực tiếp (không cần Redis/Celery).

Sử dụng:
    python run_translate.py "https://v.douyin.com/aRYGc8_w0BM/"
    python run_translate.py "https://v.douyin.com/aRYGc8_w0BM/" --cookies-from-browser chrome
    python run_translate.py "https://v.douyin.com/aRYGc8_w0BM/" --cookies cookies.txt

Output: storage/jobs/<job_id>/output/output.mp4
"""

import asyncio
import sys
import uuid
import time
import argparse
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Dịch video Douyin sang tiếng Việt")
    parser.add_argument("url", help="Douyin video URL")
    parser.add_argument("--cookies-from-browser", dest="browser", 
                        help="Lấy cookies từ browser (chrome, firefox, edge, opera)")
    parser.add_argument("--cookies", dest="cookies_file",
                        help="Path tới file cookies.txt")
    parser.add_argument("--whisper-model", default="base",
                        help="Whisper model (tiny, base, small, medium, large-v3). Default: base")
    args = parser.parse_args()

    url = args.url
    job_id = str(uuid.uuid4())[:8]
    work_dir = Path("storage/jobs") / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Douyin Video Translator - Direct Pipeline")
    print(f"{'='*60}")
    print(f"  URL: {url}")
    print(f"  Job ID: {job_id}")
    print(f"  Work dir: {work_dir}")
    print(f"{'='*60}\n")

    # Import services
    from app.services.downloader import VideoDownloader, InvalidURLError, VideoNotFoundError, NetworkError
    from app.services.audio_extractor import AudioExtractor, AudioExtractorError
    from app.services.vocal_isolator import VocalIsolator
    from app.services.speech_recognizer import SpeechRecognizer, SpeechRecognitionError
    from app.services.translator import Translator, EmptyTextError, TranslationError
    from app.services.voice_synthesizer import VoiceSynthesizer, VoiceSynthesizerError
    from app.services.video_composer import VideoComposer, VideoComposerError

    start_time = time.time()

    # Step 1: Download video
    print("[1/7] 📥 Downloading video...")
    
    import yt_dlp
    from app.services.downloader import VideoDownloader, InvalidURLError, VideoNotFoundError, NetworkError
    from app.models.pipeline import DownloadResult
    from app.models.job import VideoInfo

    downloader = VideoDownloader()

    if not downloader.validate_url(url):
        print(f"  ❌ URL không hợp lệ: {url}")
        print("  URL phải thuộc miền douyin.com (ví dụ: https://v.douyin.com/xxx)")
        sys.exit(1)

    # Download with cookies support (bypass Douyin cookie requirement)
    output_template = str(work_dir / "original.%(ext)s")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }

    if args.browser:
        ydl_opts["cookiesfrombrowser"] = (args.browser,)
        print(f"  🍪 Using cookies from: {args.browser}")
    elif args.cookies_file:
        ydl_opts["cookiefile"] = args.cookies_file
        print(f"  🍪 Using cookies file: {args.cookies_file}")
    else:
        print("  ⚠️  No cookies specified. Douyin may require cookies.")
        print("     Tip: Use --cookies-from-browser chrome")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                print("  ❌ Video không tồn tại hoặc đã bị xóa")
                sys.exit(1)
    except yt_dlp.utils.DownloadError as e:
        print(f"  ❌ Download error: {e}")
        print("\n  💡 Thử dùng: python run_translate.py URL --cookies-from-browser chrome")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        sys.exit(1)

    # Find the downloaded file
    video_path = work_dir / "original.mp4"
    if not video_path.exists():
        mp4_files = list(work_dir.glob("original.*"))
        if mp4_files:
            video_path = mp4_files[0]
        else:
            print("  ❌ Downloaded file not found")
            sys.exit(1)

    duration = info.get("duration") or 0.0
    filesize = video_path.stat().st_size
    width = info.get("width") or 0
    height = info.get("height") or 0
    resolution = f"{width}x{height}" if width and height else "unknown"
    title = info.get("title", "Unknown")

    print(f"  ✅ Downloaded: {video_path}")
    print(f"     Title: {title}")
    print(f"     Duration: {duration:.1f}s")
    print(f"     Resolution: {resolution}")
    print(f"     Size: {filesize / 1024 / 1024:.1f} MB")

    # Step 2: Extract audio
    print("\n[2/7] 🎵 Extracting audio...")
    extractor = AudioExtractor()
    try:
        audio_path = extractor.extract(video_path, work_dir)
        print(f"  ✅ Audio extracted: {audio_path}")
    except AudioExtractorError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # Step 3: Isolate vocals
    print("\n[3/7] 🎤 Isolating vocals (Demucs - may take a while)...")
    isolator = VocalIsolator()
    isolation_result = isolator.isolate(audio_path, work_dir)
    print(f"  ✅ Vocals: {isolation_result.vocals_path}")
    print(f"     Background: {isolation_result.background_path}")

    # Step 4: Speech recognition
    print(f"\n[4/7] 🗣️ Recognizing speech (Whisper {args.whisper_model} - first run downloads model)...")
    recognizer = SpeechRecognizer(model_name=args.whisper_model)
    try:
        transcription = recognizer.recognize(isolation_result.vocals_path)
        print(f"  ✅ Recognized {len(transcription.segments)} segments")
        print(f"     Language: {transcription.language}")
        print(f"     Confidence: {transcription.confidence:.2f}")
        print(f"     Text preview: {transcription.full_text[:100]}...")
    except SpeechRecognitionError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # Step 5: Translate
    print("\n[5/7] 🌐 Translating to Vietnamese...")
    translator = Translator()
    try:
        translation = translator.translate(transcription)
        print(f"  ✅ Translated {len(translation.segments)} segments")
        print(f"     Vietnamese preview: {translation.full_text_translated[:100]}...")
    except EmptyTextError as e:
        print(f"  ❌ {e.message}")
        sys.exit(1)
    except TranslationError as e:
        print(f"  ❌ {e.message}")
        sys.exit(1)

    # Step 6: Voice synthesis
    print("\n[6/7] 🔊 Synthesizing Vietnamese voice (edge-tts)...")
    synthesizer = VoiceSynthesizer()
    try:
        synthesis = asyncio.run(synthesizer.synthesize(translation, work_dir))
        print(f"  ✅ Vietnamese audio: {synthesis.audio_path}")
        print(f"     Segments synthesized: {len(synthesis.segment_audios)}")
    except VoiceSynthesizerError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # Step 7: Compose video
    print("\n[7/7] 🎬 Composing final video...")
    composer = VideoComposer()
    output_dir = work_dir / "output"
    try:
        output_path = composer.compose(
            video_path=video_path,
            vietnamese_audio=synthesis.audio_path,
            background_audio=isolation_result.background_path,
            output_dir=output_dir,
            background_volume=0.2,
        )
        elapsed = time.time() - start_time
        print(f"  ✅ Output video: {output_path}")
        print(f"\n{'='*60}")
        print(f"  🎉 DONE! Total time: {elapsed:.1f}s")
        print(f"  📁 Output: {output_path.absolute()}")
        print(f"{'='*60}")
    except VideoComposerError as e:
        print(f"  ❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
