# =============================================================================
# whisper_ui.py — NiceGUI Web-Konfigurationsoberfläche für Whisper
# UI/UX Shadcn Edition V4 (Absolute Separation)
# =============================================================================

import asyncio
import os
from pathlib import Path
from nicegui import ui, app
import sounddevice as sd
import config_rw
import process_manager
import refine

PROJECT_DIR = Path(__file__).parent
VENV_PYTHON = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
BATCH_SCRIPT = PROJECT_DIR / "batch_transcriber.py"

MODEL_OPTIONS = {
    "large-v3": "large-v3 (~3 GB)",
    "deepdml/faster-whisper-large-v3-turbo-ct2": "large-v3-turbo (~1.6 GB) [Empfohlen]",
    "Primeline/whisper-large-v3-turbo-german": "large-v3-turbo-DE (~1.6 GB)",
    "primeline/whisper-large-v3-german": "large-v3-german (~3 GB)",
    "MR-Eder/faster-whisper-large-v3-turbo-german": "MR-Eder turbo-DE",
}

COMPUTE_OPTIONS = {"float16": "float16", "bfloat16": "bfloat16"}
DEVICE_OPTIONS = {"cuda": "GPU (CUDA)", "cpu": "CPU"}
LANGUAGE_OPTIONS = {"de": "Deutsch", "en": "Englisch", "auto": "Automatisch"}
STYLE_OPTIONS = {"standard": "Standard (Sauber, Interpunktion)", "code": "Technical / Code", "raw": "Roh (ohne Satzzeichen)"}
REFINE_TARGET_OPTIONS = {
    "en": "Englisch",
    "de": "Deutsch",
    "fr": "Französisch",
    "es": "Spanisch",
    "it": "Italienisch",
    "pt": "Portugiesisch",
    "nl": "Niederländisch",
    "pl": "Polnisch",
    "ja": "Japanisch",
    "zh": "Chinesisch",
}

class _ToggleBridge:
    """
    Adapter, damit ein ui.select-Element wie ein Toggle gelesen werden kann.
    refs[…]["toggle"].value == (select.value == true_value)
    Verwendet für den Modus-Dropdown, der semantisch REFINE_TRANSLATE setzt.
    """
    def __init__(self, sel, true_value):
        self._sel = sel
        self._true_value = true_value

    @property
    def value(self) -> bool:
        return self._sel.value == self._true_value


def get_audio_devices() -> dict:
    devices = {"none": "System-Standard"}
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices[str(i)] = f"{i}: {dev['name']}"
    except Exception: pass
    return devices

def config_item(title: str, build_widget, tip: str = ""):
    with ui.column().classes("w-full gap-1 p-3 bg-zinc-900/70 border border-zinc-800/60 rounded-md justify-start"):
        with ui.row().classes("items-center gap-1 w-full"):
            ui.label(title).classes("text-zinc-500 text-[10px] font-bold uppercase tracking-wider flex-1")
            if tip:
                with ui.icon("info_outline").classes("text-zinc-600 text-xs cursor-help"):
                    ui.tooltip(tip).classes("text-xs max-w-xs")
        with ui.element("div").classes("w-full"): build_widget()

@ui.page("/")
async def index():
    cfg = config_rw.read_config()
    audio_devices = get_audio_devices()
    vocab_list: list = list(cfg.get("CUSTOM_VOCABULARY", []))
    refs = {}

    ui.add_head_html("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      body, .nicegui-content { background-color: #050505 !important; color: #fff; font-family: 'Inter', sans-serif; margin: 0; padding: 0 !important; max-width: 100vw; height: 100vh; overflow: hidden; }
      ::-webkit-scrollbar { width: 5px; height: 5px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: #27272a; border-radius: 3px; }
      ::-webkit-scrollbar-thumb:hover { background: #3f3f46; }

      /* === EMERALD COLOR SYSTEM — Kill all Quasar blue === */
      .q-btn--standard.bg-primary, .q-btn[class*="bg-primary"] { background: #059669 !important; }
      .q-field__control:before { border-color: #3f3f46 !important; }
      .q-field--focused .q-field__control:before, .q-field--focused .q-field__control:after { border-color: #10b981 !important; }
      .q-toggle__inner--truthy .q-toggle__thumb:after { background: #10b981 !important; }
      .q-toggle__inner--truthy .q-toggle__track { background: #10b981 !important; opacity: 0.5; }
      .q-slider__thumb { color: #10b981 !important; }
      .q-slider__track--inactive { background: #3f3f46 !important; }
      .q-slider__track--active { background: #10b981 !important; }
      .q-uploader { border: 1px dashed #3f3f46 !important; background: #0a0a0a !important; }
      .q-uploader__header { background: #10b981 !important; }
      .q-uploader__subtitle { opacity: 0.6; }
      /* Remove default blue from select/btn */
      .q-btn.text-primary { color: #10b981 !important; }
      .q-item__section--avatar .q-icon { color: #10b981 !important; }
      /* Global kill of Quasar primary blue on ALL buttons */
      .q-btn { --q-primary: #10b981 !important; }
      .q-btn:not(.bg-rose-900):not(.bg-zinc-100) .q-focus-helper { background: #10b981 !important; }
      /* Glow / focus ring kill */
      * { outline-color: #10b981 !important; }
      .q-field--focused .q-field__label { color: #10b981 !important; }
      /* Uploader add button */
      .q-uploader__add { color: #10b981 !important; }
      .q-uploader__header .q-btn { background: transparent !important; }

      /* === TABS === */
      .q-tabs { background: #080808; border-bottom: 1px solid #1a1a1a; min-height: 52px; }
      .q-tab { padding: 0 28px !important; text-transform: none !important; font-weight: 500; font-size: 0.875rem; letter-spacing: 0.3px; opacity: 0.55; transition: all 0.15s; }
      .q-tab:hover { opacity: 0.85; }
      .q-tab--active { background: #0f1a14 !important; color: #10b981 !important; opacity: 1; border-bottom: 2px solid #10b981; }
      .q-tab__indicator { display: none !important; }
      .q-field__control { border-radius: 5px !important; }
      .q-select__dropdown-icon { color: #52525b !important; }

      /* === BUTTON DESIGN SYSTEM (single source of truth) === */
      /* All interactive buttons follow: h-8, px-3, text-[11px], font-semibold, rounded-md */
      .btn-em  { height:32px; padding:0 12px; font-size:11px; font-weight:600; letter-spacing:.3px;
                 border-radius:5px; border:1px solid #10b981; color:#10b981;
                 background:transparent; transition:all .15s; cursor:pointer; white-space:nowrap; }
      .btn-em:hover { background:#10b98120; }
      .btn-ghost { height:32px; padding:0 12px; font-size:11px; font-weight:600;
                  border-radius:5px; border:1px solid #3f3f46; color:#71717a;
                  background:transparent; transition:all .15s; cursor:pointer; white-space:nowrap; }
      .btn-ghost:hover { border-color:#10b981; color:#10b981; }

      /* Canvas centering fix */
      #audio-canvas { display:block; margin:0 auto; }

      /* === PRIMARY CTA — immune to Quasar dark-mode primary === */
      .btn-cta { height:36px; padding:0 20px; font-size:13px; font-weight:700;
                 border-radius:6px; background:#f4f4f5 !important; color:#09090b !important;
                 border:none; transition:all .15s; cursor:pointer; white-space:nowrap; letter-spacing:.2px; }
      .btn-cta:hover { background:#d1fae5 !important; color:#064e3b !important; }
      .btn-cta-stop { background:#1c0a0a !important; color:#fca5a5 !important; border:1px solid #7f1d1d !important; }
      .btn-cta-stop:hover { background:#450a0a !important; }

      /* === Animierte Punkte für transient Status (Starte…, Lade Modell…, Stoppe…) === */
      .status-dots::after { content:''; display:inline-block; width:1.2em; text-align:left; animation: dots 1.2s steps(4,end) infinite; }
      @keyframes dots { 0% { content:''; } 25% { content:'.'; } 50% { content:'..'; } 75% { content:'...'; } 100% { content:''; } }
      /* Recording-Pulse */
      .dot-pulse { animation: pulse 1.6s ease-in-out infinite; }
      @keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.55; transform:scale(1.25); } }
    </style>
    <script>
    let audioCtx; let analyser; let dataArray; let canvasCtx; let animId; let isVisActive = false;
    let micStream = null;
    async function initAudioVisualizer() {
        if(isVisActive) return;
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;                 // war 128 — mehr bins, feinere Auflösung
            analyser.smoothingTimeConstant = 0.5;   // war 0.8 — reagiert direkter auf Lautstärke
            const source = audioCtx.createMediaStreamSource(micStream); source.connect(analyser);
            const canvas = document.getElementById('audio-canvas');
            if(canvas) { canvasCtx = canvas.getContext('2d'); dataArray = new Uint8Array(analyser.frequencyBinCount); isVisActive=true; drawWave(); }
        } catch(err) { console.warn('Mic access visualization denied by user.'); }
    }
    function stopAudioVisualizer() {
        isVisActive = false;
        if(animId) { cancelAnimationFrame(animId); animId = null; }
        try { if(micStream) { micStream.getTracks().forEach(t => t.stop()); } } catch(e) {}
        micStream = null;
        try { if(audioCtx) { audioCtx.close(); } } catch(e) {}
        audioCtx = null; analyser = null; dataArray = null;
        const canvas = document.getElementById('audio-canvas');
        if(canvas && canvas.getContext) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
    function drawWave() {
        if(!isVisActive) { animId = null; return; }
        const canvas = document.getElementById('audio-canvas');
        if(!canvas || !canvas.isConnected) {
            // Canvas wurde vom DOM entfernt (z.B. Tab-Wechsel) — Loop pausieren,
            // beim nächsten init wird er sauber neu gestartet.
            isVisActive = false; animId = null; return;
        }
        animId = requestAnimationFrame(drawWave);
        if(!canvasCtx) canvasCtx = canvas.getContext('2d');
        analyser.getByteFrequencyData(dataArray);
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        canvasCtx.fillStyle = '#10b981';

        const numBars = 32;
        const gap = 2;
        const barWidth = (canvas.width - gap * numBars) / numBars;
        const activeRange = Math.floor(dataArray.length * 0.6);

        let x = 0;
        for(let i = 0; i < numBars; i++) {
            const binIdx = Math.floor(i * (activeRange / numBars));
            // Normieren auf 0..1, dann auf Canvas-Höhe mit Boost-Faktor 1.4 für sichtbare Reaktion
            const norm = dataArray[binIdx] / 255;
            const barHeight = Math.max(3, norm * canvas.height * 1.4);
            const y = (canvas.height - barHeight) / 2;
            canvasCtx.fillRect(x, y, barWidth, barHeight);
            x += barWidth + gap;
        }
    }
    async function fallbackCopyTextToClipboard(text) {
        var textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.top = "0"; textArea.style.left = "0"; textArea.style.position = "fixed";
        document.body.appendChild(textArea);
        textArea.focus(); textArea.select();
        try { document.execCommand('copy'); } catch (err) { }
        document.body.removeChild(textArea);
    }
    function copyToClipboard(text) {
        if (!navigator.clipboard) { fallbackCopyTextToClipboard(text); return; }
        navigator.clipboard.writeText(text);
    }
    </script>
    """)

    async def save_config():
        device_val = refs.get("input_device", {}).get("sel")
        device_index = None
        if device_val and device_val.value and device_val.value != "none":
            try: device_index = int(device_val.value)
            except ValueError: pass

        updates = {
            "MODEL_SIZE": refs["model"]["sel"].value,
            "COMPUTE_TYPE": refs["compute"]["sel"].value,
            "DEVICE": refs["device"]["sel"].value,
            "GPU_DEVICE_INDEX": int(refs["gpu_idx"]["input"].value),
            "LANGUAGE": refs["lang"]["sel"].value,
            "BEAM_SIZE": int(refs["beam"]["sel"].value),
            "INITIAL_PROMPT_EXTRA": refs["prompt"]["area"].value,
            "TRANSCRIPTION_STYLE_PRESET": refs["style"]["sel"].value,
            "CUSTOM_VOCABULARY": [kw.strip() for kw in refs["vocab"]["area"].value.split(",") if kw.strip()],
            "INPUT_DEVICE_INDEX": device_index,
            "VAD_ENABLED": refs["vad"]["toggle"].value,
            "SILERO_SENSITIVITY": round(refs["silero"]["slider"].value, 2),
            "MIN_LENGTH_OF_RECORDING": round(refs["min_len"]["slider"].value, 1),
            "PRE_RECORDING_BUFFER_DURATION": round(refs["pre_buf"]["slider"].value, 1),
            "POST_SPEECH_SILENCE_DURATION": round(refs["post_silence"]["slider"].value, 1),
            "TYPE_INTO_CURSOR": refs["cursor"]["toggle"].value,
            # TASK bleibt als Whisper-Engine-Setting nominell erhalten,
            # wird aber nicht mehr an den Recorder durchgereicht.
            # Modus-Dropdown steuert jetzt REFINE_TRANSLATE.
            "TASK": "transcribe",
            "REFINE_MODEL": refs["refine_model"]["sel"].value or "",
            "REFINE_ENDPOINT": refs["refine_endpoint"]["input"].value.strip() or "http://localhost:1234/v1/chat/completions",
            "REFINE_TRANSLATE": refs["refine_translate"]["toggle"].value,
            "REFINE_TARGET_LANGUAGE": refs["refine_target"]["sel"].value,
            "REFINE_STRIP_FILLERS": refs["refine_filler"]["toggle"].value,
            "REFINE_BACKTRACK": refs["refine_backtrack"]["toggle"].value,
            "REFINE_TIMEOUT_S": float(refs["refine_timeout"]["input"].value),
        }

        # Master-Switch automatisch mit-aktivieren wenn irgendein Refine-Modus an ist.
        any_refine_mode = (
            updates["REFINE_TRANSLATE"]
            or updates["REFINE_STRIP_FILLERS"]
            or updates["REFINE_BACKTRACK"]
        )
        master_explicit = refs["refine_enabled"]["toggle"].value
        if any_refine_mode and not master_explicit:
            refs["refine_enabled"]["toggle"].value = True
            ui.notify(
                "Refine-Layer wurde mit-aktiviert, weil mindestens ein Modus eingeschaltet ist.",
                type="info", color="sky-600",
            )
        updates["REFINE_ENABLED"] = master_explicit or any_refine_mode
        try:
            config_rw.write_config(updates)
        except Exception as e:
            ui.notify(f"Fehler: {e}", type="negative", color="red-600")
            return

        if process_manager.is_running():
            ui.notify("Einstellungen gespeichert – Transcriber wird neu gestartet…", type="info", color="sky-600")
            await trigger_restart()
        else:
            ui.notify("Einstellungen gespeichert.", type="positive", color="emerald-600")

    async def trigger_restart():
        ui.notify("Startet Transcriber Prozessbaum neu...", type="info", color="sky-600")
        was_running = process_manager.is_running()
        if was_running:
            await process_manager.stop_transcription()
            # Auf sauberes Prozess-Ende polleN statt fixed sleep.
            for _ in range(60):  # max 3 s
                if not process_manager.is_running():
                    break
                await asyncio.sleep(0.05)
        if was_running:
            await process_manager.start_transcription()
        update_status()

    # --- MAIN LAYOUT (Single Column) ---
    with ui.column().classes("w-full h-full p-0 flex flex-col"):
        
        # 1. HEADER (Extremely Thin & Minimal)
        with ui.row().classes("w-full h-10 items-center justify-between px-6 bg-[#050505] shrink-0 border-b border-zinc-900"):
            with ui.row().classes("items-center gap-2"):
                ui.element("div").classes("w-2 h-2 rounded-full bg-zinc-100")
                ui.label("Whisper AI").classes("text-zinc-300 text-xs font-bold tracking-tight uppercase")
            with ui.row().classes("items-center gap-3"):
                restart_btn = ui.button(icon="refresh", on_click=trigger_restart).classes("btn-ghost shadow-none").props("flat dense")
                refs["restart_btn"] = restart_btn
                ui.button("Speichern", on_click=save_config, icon="save").classes("btn-em shadow-none").props("flat")

        # 2. THE GLOBAL TOP TABS
        with ui.tabs().classes("w-full") as global_tabs:
            tab_live = ui.tab('Live-Transkription', icon='mic')
            tab_batch = ui.tab('Dateiverarbeitung', icon='folder')
            tab_sys = ui.tab('Systemkonfiguration', icon='settings')

        # 3. TAB PANELS (Full Size)
        with ui.tab_panels(global_tabs, value=tab_live).classes("w-full flex-1 bg-[#050505] p-0"):
            
            # ======== TAB 1: LIVE TRANSCRIPTION ========
            with ui.tab_panel(tab_live).classes("w-full h-full p-0 flex flex-col items-center"):
                with ui.column().classes("w-full max-w-7xl h-full flex flex-col"):
                    
                    # Live Header: Truly centered visualizer with absolute controls
                    with ui.row().classes("w-full py-4 px-6 items-center justify-center shrink-0 border-b border-zinc-900/70 relative"):
                        with ui.row().classes("absolute left-6 items-center gap-3"):
                            async def toggle():
                                if process_manager.is_running():
                                    ui.run_javascript("stopAudioVisualizer();")
                                    await process_manager.stop_transcription()
                                else:
                                    await process_manager.start_transcription()
                                    ui.run_javascript("initAudioVisualizer();")
                                update_status()

                            main_btn = ui.button("Mikrofon aktivieren", icon="mic", on_click=toggle).classes("btn-cta shadow-none")
                            refs["main_btn"] = main_btn
                            
                            status_dot = ui.element("div").classes("w-2 h-2 rounded-full bg-zinc-600")
                            status_label = ui.label("Offline").classes("text-zinc-500 text-xs font-semibold uppercase tracking-widest")

                        # The centered piece
                        with ui.element("div").classes("h-10 w-64 bg-zinc-950/80 rounded border border-zinc-800/60 flex items-center justify-center overflow-hidden"):
                            ui.html('<canvas id="audio-canvas" width="240" height="28"></canvas>')

                    # Live Body: Feed links (flex-1) + System-Log rechts (w-80 sidebar)
                    card_texts = []
                    with ui.row().classes("w-full flex-1 flex-nowrap gap-0 min-h-0"):

                        # === LINKS: Diktierter Feed ===
                        with ui.column().classes("flex-1 min-w-0 h-full"):
                            with ui.row().classes("w-full px-6 pt-4 pb-2 justify-between items-center shrink-0"):
                                ui.label("Diktierter Feed").classes("text-zinc-600 font-semibold tracking-widest text-[10px] uppercase")
                                def copy_all():
                                    all_txt = "\n".join(card_texts)
                                    if all_txt:
                                        ui.run_javascript(f"copyToClipboard({repr(all_txt)});")
                                        ui.notify("Feed kopiert!", type="positive", color="emerald-600", position="bottom-right")
                                ui.button("Alles kopieren", icon="content_copy", on_click=copy_all).classes("btn-em shadow-none").props("flat")

                            feed_container = ui.scroll_area().classes("w-full flex-1 px-6 pb-2")
                            def create_card(text):
                                if not text.strip(): return
                                card_texts.append(text)
                                with feed_container:
                                    with ui.row().classes("w-full border border-zinc-800/70 bg-zinc-900/40 rounded-md p-4 mb-2 items-start justify-between hover:border-zinc-700 transition-colors"):
                                        ui.label(text).classes("text-zinc-200 text-sm flex-1 leading-relaxed")
                                        ui.button("Kopieren", icon="content_copy", on_click=lambda t=text: ui.run_javascript(f"copyToClipboard({repr(t)});")).classes("btn-em shadow-none ml-4 flex-shrink-0").props("flat")

                        # === RECHTS: System-Log Sidebar (immer sichtbar) ===
                        with ui.column().classes("w-80 shrink-0 h-full border-l border-zinc-900 bg-zinc-950 px-4 py-3 gap-2"):
                            with ui.row().classes("w-full items-center justify-between shrink-0"):
                                ui.label("System-Log").classes("text-zinc-500 text-[10px] font-bold tracking-wider uppercase")
                                ui.icon("terminal").classes("text-zinc-700 text-sm")
                            log_area = ui.log(max_lines=300).classes(
                                "w-full flex-1 bg-transparent text-zinc-500 font-mono text-[11px] border-none leading-relaxed"
                            )
                            def process_log(line):
                                if line.startswith("__TRANSCRIPT__:"):
                                    create_card(line.replace("__TRANSCRIPT__:", "").strip())
                                else:
                                    log_area.push(line)
                            for line in process_manager.get_log_buffer(): process_log(line)
                            process_manager.on_new_line(process_log)


            # ======== TAB 2: DATEIVERARBEITUNG ========
            with ui.tab_panel(tab_batch).classes("w-full h-full overflow-y-auto"):
                with ui.column().classes("w-full max-w-3xl mx-auto px-6 py-6 gap-4"):
                    ui.label("Batch-Verarbeitung").classes("text-lg font-semibold tracking-tight text-zinc-100")
                    ui.label("Audio- und Videodateien in die Warteschlange legen und asynchron transkribieren.").classes("text-zinc-500 text-xs mb-4")

                    batch_files = []
                    b_list = ui.column().classes("w-full gap-1")

                    def update_flist():
                        b_list.clear()
                        for f in batch_files:
                            with b_list:
                                with ui.row().classes("w-full items-center justify-between border border-zinc-800/60 bg-zinc-900/40 rounded-md px-3 py-2"):
                                    ui.label(os.path.basename(f)).classes("text-xs font-medium text-zinc-300 truncate flex-1")
                                    ui.button(icon="close", on_click=lambda path=f: (batch_files.remove(path), update_flist())).classes("btn-ghost shadow-none w-6 h-6 p-0").props("flat dense")

                    async def handle_upload(e):
                        try:
                            upload_dir = PROJECT_DIR / "99_archive" / "uploads"
                            upload_dir.mkdir(parents=True, exist_ok=True)
                            fname = e.file.name
                            out_path = upload_dir / fname
                            await e.file.save(out_path)
                            safe_path = str(out_path)
                            if safe_path not in batch_files:
                                batch_files.append(safe_path)
                            update_flist()
                            ui.notify(f"{fname} hinzugefügt.", type="positive", color="emerald-600")
                        except Exception as ex:
                            ui.notify(f"Fehler beim Upload: {ex}", type="negative", color="red-600")

                    # Compact upload area
                    ui.upload(on_upload=handle_upload, multiple=True, label="Dateien wählen oder hier ablegen (MP3, WAV, MP4, M4A)"
                              ).classes("w-full rounded-md").props("dark flat accept='.mp3,.wav,.mp4,.m4a,.ogg,.flac' color=transparent")

                    # Action row: right-aligned compact start button
                    with ui.row().classes("w-full justify-end pt-2"):
                        async def run_batch():
                            if not batch_files:
                                ui.notify("Keine Dateien in der Warteschlange.", type="warning", color="amber-600")
                                return
                            ui.notify(f"Batch gestartet ({len(batch_files)} Dateien)...", type="info", color="sky-700")
                            try:
                                proc = await asyncio.create_subprocess_exec(
                                    str(VENV_PYTHON), str(BATCH_SCRIPT), *batch_files,
                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                    cwd=str(PROJECT_DIR), creationflags=0x00000200
                                )
                                while True:
                                    line_bytes = await proc.stdout.readline()
                                    if not line_bytes: break
                                    try: line = line_bytes.decode('utf-8', errors='replace').rstrip()
                                    except: line = str(line_bytes)
                                    if line.startswith("__BATCH_RES__:"):
                                        _, data = line.split(":", 1)
                                        name, raw = data.split("|", 1)
                                        text = raw.replace("<NL>", "\n")  # decode escaped newlines
                                        with b_res:
                                            with ui.column().classes("w-full p-4 border border-zinc-800/60 bg-zinc-900/40 rounded-md mt-2"):
                                                ui.label(os.path.basename(name)).classes("font-semibold text-emerald-400 text-xs tracking-wide mb-1")
                                                ui.html(f'<p style="white-space:pre-wrap;line-height:1.75;font-size:13px;color:#d4d4d8;margin:0">{text}</p>')
                                                ui.button("Kopieren", icon="content_copy", on_click=lambda t=text: ui.run_javascript(f"copyToClipboard({repr(t)});")).classes("btn-em shadow-none mt-2").props("flat")
                                    else:
                                        b_log.push(line)
                                await proc.wait()
                                batch_files.clear(); update_flist()
                                ui.notify("Batch abgeschlossen.", type="positive", color="emerald-600")
                            except Exception as ex:
                                ui.notify(f"Subprocess-Fehler: {ex}", type="negative", color="red-600")

                        ui.button("Warteschlange starten", icon="play_arrow", on_click=run_batch).classes("btn-em shadow-none").props("flat")

                    ui.separator().classes("my-4 border-zinc-800")
                    ui.label("Live Batch-Log").classes("text-[10px] font-bold text-zinc-600 uppercase tracking-wider")
                    b_log = ui.log(max_lines=50).classes("w-full h-36 bg-[#070708] text-zinc-500 font-mono text-[11px] border border-zinc-900/80 rounded-md p-3 mt-1")
                    b_res = ui.column().classes("w-full mt-2")




            with ui.tab_panel(tab_sys).classes("w-full h-full overflow-y-auto"):
                with ui.row().classes("w-full max-w-[1600px] mx-auto px-6 py-6 gap-4 items-stretch flex-nowrap"):

                    # === LEFT SIDEBAR — Fachbegriffe ===
                    with ui.column().classes("w-72 shrink-0 gap-1 p-3 bg-zinc-900/70 border border-zinc-800/60 rounded-md self-stretch min-h-[640px]"):
                        ui.label("Eigene Fachbegriffe (Vokabular)").classes("text-zinc-500 text-[10px] font-bold uppercase tracking-wider")
                        ui.label("Komma-getrennt. Hilft Whisper, Namen und Fachwörter korrekt zu erkennen.").classes("text-zinc-600 text-[10px] leading-snug mb-1")
                        vocab_str = ", ".join(cfg.get("CUSTOM_VOCABULARY", []))
                        vocab_area = ui.textarea(placeholder="Begriff1, Begriff2, ...", value=vocab_str).classes("w-full flex-1").props("dark outlined input-class='resize-none h-full' input-style='min-height:100%'")
                        refs["vocab"] = {"area": vocab_area}

                    # === CENTER — Einstellungen Grid ===
                    with ui.column().classes("flex-1 min-w-0 gap-4"):
                        ui.label("Einstellungen & Engine Parameter").classes("text-lg font-semibold tracking-tight text-zinc-100 mb-2")
                        with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 w-full"):
                            def build_model():
                                sel = ui.select(options=MODEL_OPTIONS, value=cfg.get("MODEL_SIZE", "large-v3")).classes("w-full").props("dark outlined")
                                refs["model"] = {"sel": sel}
                            config_item("Model Auswahl", build_model, "Whisper-Modell. Größere Modelle = bessere Qualität, mehr VRAM, langsamere Verarbeitung.")

                            def buildStyle():
                                sel = ui.select(options=STYLE_OPTIONS, value=cfg.get("TRANSCRIPTION_STYLE_PRESET", "standard")).classes("w-full").props("dark outlined")
                                refs["style"] = {"sel": sel}
                            config_item("KI-Stil & Formulierung", buildStyle, "Standard: mit Satzzeichen. Code: technische Begriffe. Roh: unveränderte Whisper-Ausgabe ohne Formatierung.")

                            def build_sil():
                                sl = ui.slider(min=1.0, max=5.0, step=0.1, value=cfg.get("POST_SPEECH_SILENCE_DURATION", 3.0)).props("color=emerald-500 dark snap label")
                                refs["post_silence"] = {"slider": sl}
                            config_item("Satz-Cutoff / Denkpause (s)", build_sil, "Stille nach einem Satz, bevor transkribiert wird. Niedrig = schneller, Hoch = vollständigere Sätze.")

                            def build_vad():
                                t = ui.switch("Enable Silero VAD", value=cfg.get("VAD_ENABLED", True)).classes("text-sm text-zinc-300").props("color=emerald-500 dark")
                                refs["vad"] = {"toggle": t}
                            config_item("Voice Activity Detection", build_vad, "Erkennt automatisch Sprache im Audiosignal. Empfohlen: aktiviert. Nur bei Problemen deaktivieren.")

                            def build_beam():
                                sel = ui.select(options=[1,3,5,10], value=cfg.get("BEAM_SIZE", 5)).classes("w-full").props("dark outlined")
                                refs["beam"] = {"sel": sel}
                            config_item("Beam Size (Präzision)", build_beam, "Höher = genauere Transkription, aber langsamer. 1 = schnellste Echtzeit, 5 = Kompromiss, 10 = max. Qualität.")

                            def build_lang():
                                sel = ui.select(options=LANGUAGE_OPTIONS, value=cfg.get("LANGUAGE", "de")).classes("w-full").props("dark outlined")
                                refs["lang"] = {"sel": sel}
                            config_item("Erzwungene Sprache", build_lang, "Sprache der Audioeingabe. 'Automatisch' erkennt die Sprache, ist aber etwas langsamer.")

                            def build_device():
                                sel = ui.select(options=DEVICE_OPTIONS, value=cfg.get("DEVICE", "cuda")).classes("w-full").props("dark outlined")
                                refs["device"] = {"sel": sel}
                            config_item("Hardware Device", build_device, "GPU (CUDA) ist deutlich schneller. CPU funktioniert ohne Grafikkarte – dann Modell 'small' oder 'medium' empfohlen.")

                            def build_compute():
                                sel = ui.select(options=COMPUTE_OPTIONS, value=cfg.get("COMPUTE_TYPE", "float16")).classes("w-full").props("dark outlined")
                                refs["compute"] = {"sel": sel}
                            config_item("Compute Datentyp", build_compute, "float16: Standard für NVIDIA GPU. bfloat16: Alternative für RTX 5000er (Blackwell). CPU: int8 empfohlen.")

                            def build_gpu():
                                inp = ui.number(value=cfg.get("GPU_DEVICE_INDEX", 0), format="%.0f", min=0).classes("w-full").props("dark outlined")
                                refs["gpu_idx"] = {"input": inp}
                            config_item("GPU Index-ID", build_gpu, "Bei mehreren Grafikkarten: Index der gewünschten GPU. Standard = 0 (erste GPU).")

                            def build_mic():
                                cur = str(cfg.get("INPUT_DEVICE_INDEX")) if cfg.get("INPUT_DEVICE_INDEX") is not None else "none"
                                sel = ui.select(options=audio_devices, value=cur if cur in audio_devices else "none").classes("w-full").props("dark outlined")
                                refs["input_device"] = {"sel": sel}
                            config_item("Input Mikrofon", build_mic, "Mikrofon für die Aufnahme. 'System-Standard' übernimmt das in Windows eingestellte Gerät.")

                            def build_silero():
                                sl = ui.slider(min=0.0, max=1.0, step=0.05, value=cfg.get("SILERO_SENSITIVITY", 0.4)).props("color=emerald-500 dark snap label")
                                refs["silero"] = {"slider": sl}
                            config_item("VAD Sensibilität", build_silero, "Empfindlichkeit der Spracherkennung. Hoch = reagiert auf leise Geräusche, Niedrig = nur deutliche Sprache.")

                            def build_min():
                                sl = ui.slider(min=0.1, max=3.0, step=0.1, value=cfg.get("MIN_LENGTH_OF_RECORDING", 0.5)).props("color=emerald-500 dark snap label")
                                refs["min_len"] = {"slider": sl}
                            config_item("Min. Audio-Länge (s)", build_min, "Mindestlänge einer Aufnahme. Zu kurze Segmente werden ignoriert, um Fehltranskriptionen zu vermeiden.")

                            def build_pre_buf():
                                sl = ui.slider(min=0.0, max=5.0, step=0.1, value=cfg.get("PRE_RECORDING_BUFFER_DURATION", 1.0)).props("color=emerald-500 dark snap label")
                                refs["pre_buf"] = {"slider": sl}
                            config_item("Vorlauf-Puffer (s)", build_pre_buf, "Audio-Puffer vor Sprachbeginn. Stellt sicher, dass der Satzanfang nicht abgeschnitten wird.")

                            def build_cursor():
                                t = ui.switch("Text direkt tippen", value=cfg.get("TYPE_INTO_CURSOR", False)).classes("text-sm text-zinc-300").props("color=emerald-500 dark")
                                refs["cursor"] = {"toggle": t}
                            config_item("Input Simulation", build_cursor, "Schreibt transkribierten Text direkt an die Cursor-Position (via Zwischenablage). Vorher Fokus auf Zielfenster setzen.")

                            # === REFINE-LAYER (LM Studio) ===
                            def build_refine_enabled():
                                t = ui.switch("LLM-Refine aktivieren", value=cfg.get("REFINE_ENABLED", False)).classes("text-sm text-zinc-300").props("color=emerald-500 dark")
                                refs["refine_enabled"] = {"toggle": t}
                            config_item("Refine-Layer (LLM)", build_refine_enabled, "Master-Schalter. Schickt jedes Diktat nach Whisper durch LM Studio (lokal). Wenn aus, sind die anderen Refine-Optionen wirkungslos.")

                            def build_refine_model():
                                models = refine.list_models(cfg.get("REFINE_ENDPOINT", "http://localhost:1234/v1/chat/completions"))
                                current = cfg.get("REFINE_MODEL", "")
                                # Aktuelles Modell ergänzen, falls (noch) nicht in der Liste
                                if current and current not in models:
                                    models.insert(0, current)
                                with ui.row().classes("w-full items-center gap-1 flex-nowrap"):
                                    sel = ui.select(options=models or [current] if current else [], value=current, with_input=True).classes("w-full flex-1 min-w-0").props("dark outlined")
                                    refs["refine_model"] = {"sel": sel}
                                    def reload_models():
                                        new_list = refine.list_models(refs["refine_endpoint"]["input"].value or "http://localhost:1234/v1/chat/completions")
                                        cur = sel.value
                                        if cur and cur not in new_list:
                                            new_list.insert(0, cur)
                                        sel.set_options(new_list, value=cur if cur in new_list else (new_list[0] if new_list else None))
                                        ui.notify(f"{len(new_list)} Modelle geladen.", type="info", color="sky-600")
                                    ui.button(icon="refresh", on_click=reload_models).classes("btn-ghost shadow-none shrink-0").props("flat dense")
                            config_item("Refine-Modell", build_refine_model, "Modell aus LM Studio. Klick Refresh um die Liste neu zu laden.")

                            def _auto_enable_master(_evt=None):
                                # Live-Sync: wenn ein Refine-Modus eingeschaltet wird, Master mit anziehen.
                                if "refine_enabled" not in refs:
                                    return
                                any_mode = (
                                    (refs.get("refine_translate", {}).get("toggle") and refs["refine_translate"]["toggle"].value)
                                    or (refs.get("refine_filler", {}).get("toggle") and refs["refine_filler"]["toggle"].value)
                                    or (refs.get("refine_backtrack", {}).get("toggle") and refs["refine_backtrack"]["toggle"].value)
                                )
                                if any_mode and not refs["refine_enabled"]["toggle"].value:
                                    refs["refine_enabled"]["toggle"].value = True
                                    ui.notify("Refine-Layer wurde automatisch aktiviert.", type="info", color="sky-600")

                            def build_mode():
                                current = "translate" if cfg.get("REFINE_TRANSLATE", False) else "transcribe"
                                sel = ui.select(
                                    options={"transcribe": "Transkription (Original)", "translate": "Übersetzung (siehe Zielsprache)"},
                                    value=current,
                                    on_change=_auto_enable_master,
                                ).classes("w-full").props("dark outlined")
                                refs["refine_translate"] = {"toggle": _ToggleBridge(sel, "translate")}
                            config_item("Modus", build_mode, "Originalsprache lassen oder via LLM in die unten gewählte Zielsprache übersetzen. Aktiviert den Refine-Layer automatisch.")

                            def build_refine_target():
                                sel = ui.select(options=REFINE_TARGET_OPTIONS, value=cfg.get("REFINE_TARGET_LANGUAGE", "en")).classes("w-full").props("dark outlined")
                                refs["refine_target"] = {"sel": sel}
                            config_item("Zielsprache (Übersetzung)", build_refine_target, "Wirkt nur wenn Modus auf Übersetzung steht.")

                            def build_refine_filler():
                                t = ui.switch("Filler entfernen", value=cfg.get("REFINE_STRIP_FILLERS", False), on_change=_auto_enable_master).classes("text-sm text-zinc-300").props("color=emerald-500 dark")
                                refs["refine_filler"] = {"toggle": t}
                            config_item("Filler-Wörter entfernen", build_refine_filler, "ähm / also / halt / uh / um aus dem Output streichen.")

                            def build_refine_backtrack():
                                t = ui.switch("Self-Corrections auflösen", value=cfg.get("REFINE_BACKTRACK", False), on_change=_auto_enable_master).classes("text-sm text-zinc-300").props("color=emerald-500 dark")
                                refs["refine_backtrack"] = {"toggle": t}
                            config_item("Backtrack", build_refine_backtrack, "„Treffen Dienstag, ne Freitag\" → „Freitag\". LLM löst Self-Corrections auf.")

                            def build_refine_endpoint():
                                inp = ui.input(value=cfg.get("REFINE_ENDPOINT", "http://localhost:1234/v1/chat/completions")).classes("w-full").props("dark outlined")
                                refs["refine_endpoint"] = {"input": inp}
                            config_item("LM-Studio-Endpoint", build_refine_endpoint, "OpenAI-kompatible URL. Default ist LM Studio lokal auf Port 1234.")

                            def build_refine_timeout():
                                inp = ui.number(value=cfg.get("REFINE_TIMEOUT_S", 15.0), format="%.1f", min=1.0, max=120.0, step=0.5).classes("w-full").props("dark outlined")
                                refs["refine_timeout"] = {"input": inp}
                            config_item("Refine-Timeout (s)", build_refine_timeout, "Max. Wartezeit für den LLM-Call. Bei Timeout wird der Rohtext durchgereicht.")

                    # === RIGHT SIDEBAR — Custom System-Prompt ===
                    with ui.column().classes("w-72 shrink-0 gap-1 p-3 bg-zinc-900/70 border border-zinc-800/60 rounded-md self-stretch min-h-[640px]"):
                        ui.label("Custom System-Prompt").classes("text-zinc-500 text-[10px] font-bold uppercase tracking-wider")
                        ui.label("Freier Text als initial_prompt für Whisper. Steuert Stil, Tonalität oder Formatierungs-Hinweise.").classes("text-zinc-600 text-[10px] leading-snug mb-1")
                        prompt_area = ui.textarea(placeholder="Anweisungen zur Formatierung...", value=cfg.get("INITIAL_PROMPT_EXTRA", "")).classes("w-full flex-1").props("dark outlined input-class='resize-none h-full' input-style='min-height:100%'")
                        refs["prompt"] = {"area": prompt_area}


    # State → (dot-Farbe, label-Basis-Text, label-Farbe, button-Text, button-Icon,
    #           button-disabled, recording-pulse, animierte-Punkte am Label)
    _STATE_VIEW = {
        "offline":       ("bg-zinc-600",    "Offline",      "text-zinc-500",    "Mikrofon aktivieren",   "mic",             False, False, False),
        "starting":      ("bg-amber-500",   "Starte",       "text-amber-500",   "Wird gestartet",        "hourglass_empty", True,  False, True),
        "loading_model": ("bg-amber-500",   "Lade Modell",  "text-amber-500",   "Lade Modell",           "downloading",     True,  False, True),
        "ready":         ("bg-emerald-500", "Bereit",       "text-emerald-500", "Mikrofon deaktivieren", "stop",            False, False, False),
        "recording":     ("bg-emerald-500", "Aufnahme",     "text-emerald-500", "Mikrofon deaktivieren", "stop",            False, True,  False),
        "refining":      ("bg-sky-500",     "Verfeinere",   "text-sky-400",     "Mikrofon deaktivieren", "stop",            False, False, True),
        "stopping":      ("bg-amber-500",   "Stoppe",       "text-amber-500",   "Wird gestoppt",         "hourglass_empty", True,  False, True),
    }
    _ALL_DOT_COLORS = "bg-zinc-600 bg-amber-500 bg-emerald-500 bg-sky-500"
    _ALL_LABEL_COLORS = "text-zinc-500 text-amber-500 text-emerald-500 text-sky-400"

    def update_status(*_args):
        state = process_manager.get_state()
        # Wenn Prozess weg, aber State noch nicht reset (defensiv):
        if not process_manager.is_running() and state not in ("offline", "stopping"):
            state = "offline"
        view = _STATE_VIEW.get(state, _STATE_VIEW["offline"])
        dot_color, lbl_text, lbl_color, btn_text, btn_icon, btn_disabled, dot_pulse, dots_anim = view

        status_dot.classes(remove=_ALL_DOT_COLORS, add=dot_color)
        if dot_pulse:
            status_dot.classes(add="dot-pulse")
        else:
            status_dot.classes(remove="dot-pulse")

        status_label.set_text(lbl_text)
        status_label.classes(remove=_ALL_LABEL_COLORS, add=lbl_color)
        if dots_anim:
            status_label.classes(add="status-dots")
        else:
            status_label.classes(remove="status-dots")

        main_btn = refs["main_btn"]
        main_btn.set_text(btn_text)
        main_btn.props(remove="icon=mic icon=stop icon=hourglass_empty icon=downloading", add=f"icon={btn_icon}")
        if state in ("ready", "recording"):
            main_btn.classes(remove="btn-cta", add="btn-cta btn-cta-stop")
        else:
            main_btn.classes(remove="btn-cta-stop", add="btn-cta")
        main_btn.set_enabled(not btn_disabled)

        # Beim Wechsel auf offline auch den Browser-Mic-Stream sauber stoppen.
        if state == "offline":
            try:
                ui.run_javascript("stopAudioVisualizer();")
            except Exception:
                pass

    # State-Wechsel aus dem Subprozess (z. B. loading_model → ready) sofort
    # in die UI bringen.
    process_manager.on_state_change(lambda _new: update_status())

    update_status()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Whisper AI", port=8080, dark=True, reload=False, show=False)
