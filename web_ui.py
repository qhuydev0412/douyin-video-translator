"""Web UI cho Douyin Video Translator — upload video trực tiếp từ browser.

Chạy:
    python web_ui.py

Mở browser: http://localhost:8080
"""

import asyncio
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Douyin Video Translator - Web UI")

# Store job statuses in memory
jobs: dict[str, dict] = {}

STORAGE = Path("storage/jobs")
STORAGE.mkdir(parents=True, exist_ok=True)


# ─── HTML UI ────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Douyin Video Translator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; padding: 2rem; }
.container { max-width: 700px; margin: 0 auto; }
h1 { text-align: center; margin-bottom: 0.5rem; font-size: 1.8rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; color: #888; margin-bottom: 2rem; font-size: 0.9rem; }
.card { background: #1a1a2e; border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid #2a2a4a; }
.upload-zone { border: 2px dashed #444; border-radius: 8px; padding: 3rem 1rem; text-align: center; cursor: pointer; transition: all 0.3s; }
.upload-zone:hover, .upload-zone.dragover { border-color: #667eea; background: rgba(102, 126, 234, 0.05); }
.upload-zone input { display: none; }
.upload-zone p { color: #aaa; margin-top: 0.5rem; font-size: 0.85rem; }
.upload-icon { font-size: 3rem; }
.file-name { margin-top: 1rem; color: #667eea; font-weight: 500; }
.options { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
label { font-size: 0.85rem; color: #aaa; margin-bottom: 0.3rem; display: block; }
select, input[type="number"] { width: 100%; padding: 0.6rem; background: #16213e; border: 1px solid #333; border-radius: 6px; color: #e0e0e0; font-size: 0.9rem; }
.btn { width: 100%; padding: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 8px; color: white; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 1.5rem; transition: opacity 0.3s; }
.btn:hover { opacity: 0.9; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.progress-section { display: none; }
.progress-bar { width: 100%; height: 8px; background: #2a2a4a; border-radius: 4px; overflow: hidden; margin: 1rem 0; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.5s ease; width: 0%; }
.status { font-size: 0.9rem; color: #aaa; }
.step { color: #667eea; font-weight: 500; }
.log { background: #0d1117; border-radius: 8px; padding: 1rem; margin-top: 1rem; max-height: 200px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 0.8rem; line-height: 1.6; }
.log-entry { color: #8b949e; }
.log-entry.success { color: #3fb950; }
.log-entry.error { color: #f85149; }
.download-section { display: none; text-align: center; padding: 2rem; }
.download-btn { display: inline-block; padding: 1rem 2rem; background: #3fb950; border-radius: 8px; color: white; text-decoration: none; font-weight: 600; }
.error-msg { color: #f85149; margin-top: 1rem; }
</style>
</head>
<body>
<div class="container">
  <h1>🎬 Douyin Video Translator</h1>
  <p class="subtitle">Upload video tiếng Trung → Nhận video thuyết minh tiếng Việt</p>

  <div class="card" id="upload-card">
    <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
      <div class="upload-icon">📁</div>
      <p>Kéo thả video vào đây hoặc click để chọn file</p>
      <p>Hỗ trợ: MP4, MOV, AVI, MKV (tối đa 500MB)</p>
      <input type="file" id="file-input" accept="video/*">
      <div class="file-name" id="file-name"></div>
    </div>

    <div class="options">
      <div>
        <label>Whisper Model</label>
        <select id="whisper-model">
          <option value="base">base (~150MB, nhanh)</option>
          <option value="small">small (~500MB)</option>
          <option value="medium">medium (~1.5GB)</option>
          <option value="large-v3">large-v3 (~3GB, tốt nhất)</option>
        </select>
      </div>
      <div>
        <label>Âm lượng nhạc nền</label>
        <select id="bg-volume">
          <option value="0.1">10%</option>
          <option value="0.2" selected>20%</option>
          <option value="0.3">30%</option>
          <option value="0.5">50%</option>
        </select>
      </div>
    </div>

    <button class="btn" id="start-btn" onclick="startTranslation()" disabled>
      🚀 Bắt đầu dịch
    </button>
  </div>

  <div class="card progress-section" id="progress-section">
    <div class="status">
      Bước: <span class="step" id="current-step">Đang chuẩn bị...</span>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" id="progress-fill"></div>
    </div>
    <div class="status" id="progress-text">0%</div>
    <div class="log" id="log"></div>
  </div>

  <div class="card download-section" id="download-section">
    <div class="upload-icon">🎉</div>
    <h2 style="margin: 1rem 0;">Dịch video thành công!</h2>
    <a class="download-btn" id="download-link" href="#">📥 Tải video</a>
  </div>
</div>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const startBtn = document.getElementById('start-btn');
let selectedFile = null;

// Drag and drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) { handleFile(e.dataTransfer.files[0]); }
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
  if (file.size > 500 * 1024 * 1024) { alert('File quá lớn (tối đa 500MB)'); return; }
  selectedFile = file;
  fileName.textContent = `${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
  startBtn.disabled = false;
}

function addLog(msg, type = '') {
  const log = document.getElementById('log');
  const entry = document.createElement('div');
  entry.className = 'log-entry ' + type;
  entry.textContent = msg;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

async function startTranslation() {
  if (!selectedFile) return;
  
  startBtn.disabled = true;
  document.getElementById('progress-section').style.display = 'block';
  
  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('whisper_model', document.getElementById('whisper-model').value);
  formData.append('bg_volume', document.getElementById('bg-volume').value);
  
  addLog('📤 Uploading video...');
  
  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    
    if (!resp.ok) { addLog('❌ ' + data.error, 'error'); return; }
    
    addLog('✅ Upload complete, starting pipeline...', 'success');
    pollStatus(data.job_id);
  } catch (err) {
    addLog('❌ Upload failed: ' + err.message, 'error');
  }
}

async function pollStatus(jobId) {
  const interval = setInterval(async () => {
    try {
      const resp = await fetch('/api/status/' + jobId);
      const data = await resp.json();
      
      document.getElementById('current-step').textContent = data.step || 'Processing...';
      document.getElementById('progress-fill').style.width = data.progress + '%';
      document.getElementById('progress-text').textContent = data.progress + '%';
      
      if (data.logs && data.logs.length) {
        const log = document.getElementById('log');
        log.innerHTML = '';
        data.logs.forEach(l => addLog(l.msg, l.type));
      }
      
      if (data.status === 'completed') {
        clearInterval(interval);
        document.getElementById('download-section').style.display = 'block';
        document.getElementById('download-link').href = '/api/download/' + jobId;
        addLog('🎉 Hoàn tất!', 'success');
      } else if (data.status === 'failed') {
        clearInterval(interval);
        addLog('❌ ' + (data.error || 'Pipeline failed'), 'error');
        startBtn.disabled = false;
      }
    } catch (err) { /* retry */ }
  }, 2000);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    whisper_model: str = Form("base"),
    bg_volume: float = Form(0.2),
):
    """Upload video file and start translation pipeline."""
    job_id = str(uuid.uuid4())[:8]
    work_dir = STORAGE / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    video_path = work_dir / f"original{Path(file.filename).suffix}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize job state
    jobs[job_id] = {
        "status": "processing",
        "step": "Uploaded",
        "progress": 0,
        "logs": [{"msg": f"📁 File: {file.filename} ({file.size / 1024 / 1024:.1f} MB)", "type": ""}],
        "video_path": str(video_path),
        "whisper_model": whisper_model,
        "bg_volume": bg_volume,
        "output_path": None,
        "error": None,
    }

    # Start pipeline in background thread
    thread = Thread(target=run_pipeline, args=(job_id,), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get job status and progress."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job = jobs[job_id]
    return {
        "status": job["status"],
        "step": job["step"],
        "progress": job["progress"],
        "logs": job["logs"],
        "error": job["error"],
    }


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    """Download the translated video."""
    if job_id not in jobs or not jobs[job_id].get("output_path"):
        return JSONResponse({"error": "Video not found"}, status_code=404)
    output_path = Path(jobs[job_id]["output_path"])
    if not output_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"translated_{job_id}.mp4",
    )


def run_pipeline(job_id: str):
    """Run the translation pipeline synchronously in a background thread."""
    job = jobs[job_id]
    video_path = Path(job["video_path"])
    work_dir = video_path.parent
    whisper_model = job["whisper_model"]
    bg_volume = job["bg_volume"]

    def log(msg, log_type=""):
        job["logs"].append({"msg": msg, "type": log_type})

    def update(step, progress):
        job["step"] = step
        job["progress"] = progress

    try:
        from app.services.audio_extractor import AudioExtractor, AudioExtractorError
        from app.services.vocal_isolator import VocalIsolator
        from app.services.speech_recognizer import SpeechRecognizer, SpeechRecognitionError
        from app.services.translator import Translator, EmptyTextError, TranslationError
        from app.services.voice_synthesizer import VoiceSynthesizer, VoiceSynthesizerError
        from app.services.video_composer import VideoComposer, VideoComposerError

        # Step 1: Extract audio (15%)
        update("Tách âm thanh (FFmpeg)...", 5)
        log("🎵 Extracting audio from video...")
        extractor = AudioExtractor()
        audio_path = extractor.extract(video_path, work_dir)
        update("Tách âm thanh", 15)
        log("✅ Audio extracted", "success")

        # Step 2: Isolate vocals (35%)
        update("Tách giọng nói (Demucs)...", 20)
        log("🎤 Isolating vocals with Demucs...")
        isolator = VocalIsolator()
        isolation = isolator.isolate(audio_path, work_dir)
        update("Tách giọng nói", 35)
        log("✅ Vocals isolated", "success")

        # Step 3: Speech recognition (55%)
        update(f"Nhận dạng giọng nói (Whisper {whisper_model})...", 40)
        log(f"🗣️ Recognizing speech with Whisper {whisper_model}...")
        recognizer = SpeechRecognizer(model_name=whisper_model)
        transcription = recognizer.recognize(isolation.vocals_path)
        update("Nhận dạng giọng nói", 55)
        log(f"✅ Recognized {len(transcription.segments)} segments", "success")
        log(f"   Text: {transcription.full_text[:80]}...")

        # Step 4: Translate (70%)
        update("Dịch sang tiếng Việt...", 60)
        log("🌐 Translating to Vietnamese...")
        translator = Translator()
        translation = translator.translate(transcription)
        update("Dịch sang tiếng Việt", 70)
        log(f"✅ Translated: {translation.full_text_translated[:80]}...", "success")

        # Step 5: Voice synthesis (85%)
        update("Tổng hợp giọng Việt (edge-tts)...", 75)
        log("🔊 Synthesizing Vietnamese voice...")
        synthesizer = VoiceSynthesizer()
        synthesis = asyncio.run(synthesizer.synthesize(translation, work_dir))
        update("Tổng hợp giọng Việt", 85)
        log(f"✅ Synthesized {len(synthesis.segment_audios)} audio segments", "success")

        # Step 6: Compose video (100%)
        update("Ghép video...", 90)
        log("🎬 Composing final video...")
        composer = VideoComposer()
        output_dir = work_dir / "output"
        output_path = composer.compose(
            video_path=video_path,
            vietnamese_audio=synthesis.audio_path,
            background_audio=isolation.background_path,
            output_dir=output_dir,
            background_volume=bg_volume,
        )
        update("Hoàn tất! 🎉", 100)
        log(f"✅ Output: {output_path}", "success")

        job["status"] = "completed"
        job["output_path"] = str(output_path)

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        log(f"❌ Error: {e}", "error")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8080"))
    print("=" * 50)
    print("  🎬 Douyin Video Translator - Web UI")
    print(f"  Open: http://localhost:{port}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)
