"""Web UI cho Douyin Video Translator — mount lên app.main và thêm route HTML."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.responses import HTMLResponse
from app.main import app
import uvicorn

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Douyin Video Translator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; padding: 2rem; }
.container { max-width: 760px; margin: 0 auto; }
h1 { text-align: center; margin-bottom: 0.5rem; font-size: 1.8rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subtitle { text-align: center; color: #888; margin-bottom: 2rem; font-size: 0.9rem; }
.card { background: #1a1a2e; border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem; border: 1px solid #2a2a4a; }
.card h2 { font-size: 1rem; margin-bottom: 1.2rem; color: #aaa; }
label { font-size: 0.85rem; color: #aaa; margin-bottom: 0.3rem; display: block; }
input[type=text] { width: 100%; padding: 0.8rem 1rem; background: #16213e; border: 1px solid #333; border-radius: 8px; color: #e0e0e0; font-size: 0.95rem; outline: none; transition: border-color .2s; }
input[type=text]:focus { border-color: #667eea; }
input[type=text]::placeholder { color: #555; }
.btn { width: 100%; padding: 0.9rem; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 8px; color: white; font-size: 1rem; font-weight: 600; cursor: pointer; margin-top: 1.2rem; transition: opacity .2s; }
.btn:hover { opacity: .9; }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.btn-sm { width: auto; padding: 0.5rem 1.2rem; font-size: 0.85rem; margin-top: 0; border-radius: 6px; }
.btn-green { background: #3fb950; }
.btn-gray { background: #444; }
/* Progress */
.progress-bar { width: 100%; height: 8px; background: #2a2a4a; border-radius: 4px; overflow: hidden; margin: 1rem 0; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width .5s ease; width: 0%; }
.status-label { font-size: 0.9rem; color: #aaa; margin-bottom: .3rem; }
.step-label { color: #667eea; font-weight: 600; }
.log { background: #0d1117; border-radius: 8px; padding: 1rem; margin-top: 1rem; max-height: 160px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; line-height: 1.6; }
.log-entry { color: #8b949e; }
.log-entry.ok { color: #3fb950; }
.log-entry.err { color: #f85149; }
/* Segment table */
.seg-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.seg-table th { text-align: left; padding: 0.5rem 0.6rem; color: #667eea; border-bottom: 1px solid #2a2a4a; font-weight: 600; }
.seg-table td { padding: 0.45rem 0.6rem; border-bottom: 1px solid #1e1e3a; vertical-align: top; }
.seg-table tr:hover td { background: #1e1e3a; }
.time-cell { color: #666; white-space: nowrap; font-size: 0.78rem; }
.editable { background: #0d1117; border: 1px solid #333; border-radius: 4px; color: #e0e0e0; padding: 0.3rem 0.5rem; width: 100%; font-family: inherit; font-size: 0.85rem; resize: vertical; min-height: 1.8rem; }
.editable:focus { outline: none; border-color: #667eea; }
.zh-text { color: #ccc; }
/* Voice cards */
.voice-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.8rem; }
.voice-card { background: #16213e; border: 2px solid #2a2a4a; border-radius: 8px; padding: 1rem; cursor: pointer; transition: border-color .2s; }
.voice-card:hover { border-color: #667eea; }
.voice-card.selected { border-color: #764ba2; background: #1a1a3e; }
.voice-name { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem; }
.confirm-row { display: flex; justify-content: flex-end; gap: 0.7rem; margin-top: 1.2rem; }
/* Download */
.download-section { text-align: center; }
.download-btn { display: inline-block; padding: 1rem 2.5rem; background: #3fb950; border-radius: 8px; color: white; text-decoration: none; font-weight: 700; font-size: 1rem; }
.hidden { display: none !important; }
</style>
</head>
<body>
<div class="container">
  <h1>🎬 Douyin Video Translator</h1>
  <p class="subtitle">Nhập link Douyin → Preview → Confirm → Nhận video thuyết minh tiếng Việt</p>

  <!-- INPUT CARD -->
  <div class="card" id="input-card">
    <label>Link video Douyin</label>
    <input type="text" id="url-input" placeholder="https://www.douyin.com/video/..." oninput="onUrl(this.value)" />
    <button class="btn" id="start-btn" onclick="startJob()" disabled>🚀 Bắt đầu dịch</button>
  </div>

  <!-- PROGRESS CARD -->
  <div class="card hidden" id="progress-card">
    <div class="status-label">Bước: <span class="step-label" id="step-txt">Đang xử lý...</span></div>
    <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
    <div class="status-label" id="prog-pct">0%</div>
    <div class="log" id="log-box"></div>
  </div>

  <!-- TRANSLATION CHECKPOINT -->
  <div class="card hidden" id="translation-card">
    <h2>🌐 Kiểm tra bản dịch tiếng Việt</h2>
    <div style="overflow-x:auto">
      <table class="seg-table">
        <thead><tr><th>#</th><th>Thời gian</th><th>Tiếng Trung</th><th>Tiếng Việt</th></tr></thead>
        <tbody id="translation-tbody"></tbody>
      </table>
    </div>
    <div class="confirm-row">
      <button class="btn btn-sm btn-green" onclick="confirmTranslation()">✅ Xác nhận &amp; tiếp tục</button>
    </div>
  </div>

  <!-- AUDIO PREVIEW CHECKPOINT -->
  <div class="card hidden" id="audio-preview-card">
    <h2>🎧 Nghe thử &amp; chỉnh sửa trước khi ghép video</h2>
    <p style="font-size:0.82rem;color:#888;margin-bottom:1rem">Nhấn ▶ để nghe từng đoạn. Speed tự động điều chỉnh để khớp thời gian gốc. Sửa text nếu cần rồi xác nhận.</p>
    <div style="overflow-x:auto">
      <table class="seg-table">
        <thead><tr><th>#</th><th>Thời gian</th><th>Tiếng Trung</th><th>Tiếng Việt</th><th>Giọng</th><th></th></tr></thead>
        <tbody id="audio-tbody"></tbody>
      </table>
    </div>
    <div class="confirm-row">
      <button class="btn btn-sm btn-green" onclick="confirmAudio()">✅ Xác nhận &amp; ghép video</button>
    </div>
  </div>

  <!-- VOICE SELECTION CHECKPOINT -->
  <div class="card hidden" id="voice-card">
    <h2>🔊 Chọn giọng đọc tiếng Việt</h2>
    <div class="voice-grid" id="voice-grid"></div>
    <div class="confirm-row">
      <button class="btn btn-sm btn-green" id="voice-confirm-btn" onclick="confirmVoice()" disabled>✅ Chọn giọng này</button>
    </div>
  </div>

  <!-- DOWNLOAD -->
  <div class="card hidden download-section" id="download-card">
    <div style="font-size:3rem;margin-bottom:1rem">🎉</div>
    <h2 style="margin-bottom:1.5rem">Dịch video thành công!</h2>
    <a class="download-btn" id="download-link" href="#">📥 Tải video</a>
  </div>
</div>

<script>
let jobId = null;
let pollTimer = null;
let selectedVoiceId = null;

function onUrl(v) {
  document.getElementById('start-btn').disabled = !v.includes('douyin.com');
}

function fmt(s) {
  const m = Math.floor(s / 60), sec = (s % 60).toFixed(1);
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(4,'0')}`;
}

function addLog(msg, type='') {
  const box = document.getElementById('log-box');
  const d = document.createElement('div');
  d.className = 'log-entry ' + type;
  d.textContent = msg;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }
function hideCheckpoints() {
  hide('translation-card'); hide('voice-card'); hide('audio-preview-card');
}

async function startJob() {
  const url = document.getElementById('url-input').value.trim();
  if (!url) return;
  document.getElementById('start-btn').disabled = true;
  show('progress-card');
  addLog('🔗 Đang gửi yêu cầu...');

  const resp = await fetch('/api/v1/translate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url}),
  });
  const data = await resp.json();
  if (!resp.ok) { addLog('❌ ' + (data.message || 'Lỗi'), 'err'); document.getElementById('start-btn').disabled = false; return; }

  jobId = data.job_id;
  addLog(`✅ Tạo job: ${jobId}`, 'ok');
  pollTimer = setInterval(poll, 2000);
}

async function poll() {
  if (!jobId) return;
  const resp = await fetch(`/api/v1/jobs/${jobId}`);
  if (!resp.ok) return;
  const d = await resp.json();

  // Progress
  document.getElementById('step-txt').textContent = d.current_step || d.status;
  document.getElementById('prog-fill').style.width = d.progress_percent + '%';
  document.getElementById('prog-pct').textContent = d.progress_percent + '%';

  if (d.status === 'awaiting_confirmation') {
    clearInterval(pollTimer); pollTimer = null;
    hideCheckpoints();

    if (d.checkpoint_type === 'translation') {
      renderTranslation(d.preview_data?.translation_segments || []);
      show('translation-card');
    } else if (d.checkpoint_type === 'voice_selection') {
      renderVoices(d.preview_data?.voice_options || []);
      show('voice-card');
    } else if (d.checkpoint_type === 'audio_preview') {
      renderAudioPreview(d.preview_data?.audio_preview_segments || []);
      show('audio-preview-card');
    }
  } else if (d.status === 'completed') {
    clearInterval(pollTimer); pollTimer = null;
    hideCheckpoints();
    addLog('🎉 Hoàn tất!', 'ok');
    document.getElementById('download-link').href = `/api/v1/jobs/${jobId}/download`;
    show('download-card');
  } else if (d.status === 'failed') {
    clearInterval(pollTimer); pollTimer = null;
    addLog('❌ ' + (d.error?.message || 'Pipeline thất bại'), 'err');
    document.getElementById('start-btn').disabled = false;
  } else {
    // still processing
    addLog(`⏳ ${d.current_step || d.status} (${d.progress_percent}%)`);
  }
}

/* ── Translation ── */
function renderTranslation(segs) {
  const tb = document.getElementById('translation-tbody');
  tb.innerHTML = '';
  segs.forEach(s => {
    const dur = (s.end - s.start).toFixed(1);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="time-cell">${s.index + 1}</td>
      <td class="time-cell">${fmt(s.start)}<br>${fmt(s.end)}<br><span style="color:#555">${dur}s</span></td>
      <td class="zh-text">${s.original_text}</td>
      <td><textarea class="editable" rows="2" data-index="${s.index}">${s.translated_text}</textarea></td>`;
    tb.appendChild(tr);
  });
}

/* ── Audio Preview ── */
const VOICES = ['alloy','echo','fable','onyx','nova','shimmer'];
const VOICE_LABELS = {alloy:'Alloy (neutral)',echo:'Echo (male)',fable:'Fable (expressive)',onyx:'Onyx (male)',nova:'Nova (female)',shimmer:'Shimmer (female)'};

function renderAudioPreview(segs) {
  const tb = document.getElementById('audio-tbody');
  tb.innerHTML = '';
  segs.forEach(s => {
    const dur = (s.end - s.start).toFixed(1);
    const tr = document.createElement('tr');
    const voiceOpts = VOICES.map(v =>
      `<option value="${v}"${v === s.voice ? ' selected' : ''}>${VOICE_LABELS[v] || v}</option>`
    ).join('');
    tr.innerHTML = `
      <td class="time-cell">${s.index + 1}</td>
      <td class="time-cell">${fmt(s.start)}<br>${fmt(s.end)}<br><span style="color:#555">${dur}s</span></td>
      <td class="zh-text">${s.original_text}</td>
      <td><textarea class="editable" rows="2" data-seg-index="${s.index}">${s.translated_text}</textarea></td>
      <td><select class="voice-select" data-voice-index="${s.index}" style="font-size:0.78rem;padding:2px 4px;border-radius:4px;border:1px solid #444;background:#1a1a1a;color:#eee">${voiceOpts}</select></td>
      <td><button class="btn btn-sm btn-gray" style="margin-top:0;white-space:nowrap"
        onclick="playSegment(${s.index})">▶ Nghe</button></td>`;
    tb.appendChild(tr);
  });
}

function playSegment(idx) {
  const url = `/api/v1/jobs/${jobId}/segments/${idx}/audio`;
  if (window._segAudio) { window._segAudio.pause(); }
  window._segAudio = new Audio(url);
  window._segAudio.play().catch(() => addLog('⚠️ Không thể phát audio', 'err'));
}

async function confirmAudio() {
  const edits = [];
  // Collect text edits
  document.querySelectorAll('#audio-tbody textarea').forEach(el => {
    edits.push({index: parseInt(el.dataset.segIndex), translated_text: el.value, voice: null});
  });
  // Merge voice selections into the same edit entries
  document.querySelectorAll('#audio-tbody select.voice-select').forEach(el => {
    const idx = parseInt(el.dataset.voiceIndex);
    const existing = edits.find(e => e.index === idx);
    if (existing) { existing.voice = el.value; }
    else { edits.push({index: idx, translated_text: null, voice: el.value}); }
  });
  hide('audio-preview-card');
  addLog('✅ Đã xác nhận audio, đang tổng hợp lại và ghép video...', 'ok');
  show('progress-card');

  await fetch(`/api/v1/jobs/${jobId}/confirm/audio`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({edits}),
  });
  pollTimer = setInterval(poll, 2000);
}

async function confirmTranslation() {
  const edits = [];
  document.querySelectorAll('#translation-tbody textarea').forEach(el => {
    edits.push({index: parseInt(el.dataset.index), translated_text: el.value});
  });
  hide('translation-card');
  addLog('✅ Đã xác nhận bản dịch tiếng Việt, tiếp tục tổng hợp giọng...', 'ok');
  show('progress-card');

  await fetch(`/api/v1/jobs/${jobId}/confirm/translation`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({edits}),
  });
  pollTimer = setInterval(poll, 2000);
}

/* ── Voice Selection ── */
function renderVoices(options) {
  const grid = document.getElementById('voice-grid');
  grid.innerHTML = '';
  selectedVoiceId = null;
  document.getElementById('voice-confirm-btn').disabled = true;

  options.forEach(v => {
    const div = document.createElement('div');
    div.className = 'voice-card';
    div.dataset.voiceId = v.voice_id;
    div.innerHTML = `
      <div class="voice-name">${v.voice_name}</div>
      <button class="btn btn-sm btn-gray" style="margin-top:.5rem"
        onclick="playPreview(event,'${jobId}','${v.voice_id}')">▶ Nghe thử</button>`;
    div.addEventListener('click', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      selectVoice(v.voice_id);
    });
    grid.appendChild(div);
  });
}

function selectVoice(voiceId) {
  selectedVoiceId = voiceId;
  document.querySelectorAll('.voice-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.voiceId === voiceId);
  });
  document.getElementById('voice-confirm-btn').disabled = false;
}

async function playPreview(e, jid, vid) {
  e.stopPropagation();
  const url = `/api/v1/jobs/${jid}/preview/voice/${vid}`;
  const audio = new Audio(url);
  audio.play().catch(() => addLog('⚠️ Không thể phát audio preview', 'err'));
}

async function confirmVoice() {
  if (!selectedVoiceId) return;
  hide('voice-card');
  addLog(`✅ Đã chọn giọng: ${selectedVoiceId}, tiếp tục tổng hợp...`, 'ok');
  show('progress-card');

  await fetch(`/api/v1/jobs/${jobId}/confirm/voice`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({voice_id: selectedVoiceId}),
  });
  pollTimer = setInterval(poll, 2000);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print("=" * 50)
    print("  🎬 Douyin Video Translator")
    print(f"  Open: http://localhost:{port}")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=port)
