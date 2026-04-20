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

def get_audio_devices() -> dict:
    devices = {"none": "System-Standard"}
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices[str(i)] = f"{i}: {dev['name']}"
    except Exception: pass
    return devices

def config_item(title: str, build_widget):
    with ui.column().classes("w-full gap-1 p-3 bg-zinc-900/70 border border-zinc-800/60 rounded-md justify-start"):
        ui.label(title).classes("text-zinc-500 text-[10px] font-bold uppercase tracking-wider")
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
    </style>
    <script>
    let audioCtx; let analyser; let dataArray; let canvasCtx; let animId; let isVisActive = false;
    async function initAudioVisualizer() {
        if(isVisActive) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser(); analyser.fftSize = 128; analyser.smoothingTimeConstant = 0.8;
            const source = audioCtx.createMediaStreamSource(stream); source.connect(analyser);
            const canvas = document.getElementById('audio-canvas');
            if(canvas) { canvasCtx = canvas.getContext('2d'); dataArray = new Uint8Array(analyser.frequencyBinCount); isVisActive=true; drawWave(); }
        } catch(err) { console.warn('Mic access visualization denied by user.'); }
    }
    function drawWave() {
        if(!isVisActive) return;
        animId = requestAnimationFrame(drawWave);
        const canvas = document.getElementById('audio-canvas');
        if(!canvas || !canvasCtx) return;
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
            const barHeight = dataArray[binIdx] / 4;
            const y = (canvas.height - barHeight) / 2;
            canvasCtx.fillRect(x, y, barWidth, barHeight || 2);
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

    def save_config():
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
        }
        try:
            config_rw.write_config(updates)
            ui.notify("Einstellungen gespeichert.", type="positive", color="emerald-600")
            refs["restart_btn"].classes(remove="text-zinc-600", add="text-amber-500 hover:text-amber-400 bg-amber-900/20")
        except Exception as e:
            ui.notify(f"Fehler: {e}", type="negative", color="red-600")

    async def trigger_restart():
        ui.notify("Startet Transcriber Prozessbaum neu...", type="info", color="sky-600")
        was_running = process_manager.is_running()
        if was_running: await process_manager.stop_transcription()
        await asyncio.sleep(0.5)
        if was_running: await process_manager.start_transcription()
        refs["restart_btn"].classes(remove="text-amber-500 hover:text-amber-400 bg-amber-900/20", add="text-zinc-600 hover:text-zinc-300")
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
                                if process_manager.is_running(): await process_manager.stop_transcription()
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

                    # Live Middle: Card Feed
                    card_texts = []
                    with ui.row().classes("w-full px-6 pt-4 pb-2 justify-between items-center shrink-0"):
                        ui.label("Diktierter Feed").classes("text-zinc-600 font-semibold tracking-widest text-[10px] uppercase")
                        def copy_all():
                            all_txt = "\n".join(card_texts)
                            if all_txt:
                                ui.run_javascript(f"copyToClipboard({repr(all_txt)});")
                                ui.notify("Feed kopiert!", type="positive", color="emerald-600", position="bottom-right")
                        ui.button("Alles kopieren", icon="content_copy", on_click=copy_all).classes("btn-em shadow-none").props("flat")

                    feed_container = ui.scroll_area().classes("w-full flex-1 px-6 mb-2")
                    def create_card(text):
                        if not text.strip(): return
                        card_texts.append(text)
                        with feed_container:
                            with ui.row().classes("w-full border border-zinc-800/70 bg-zinc-900/40 rounded-md p-4 mb-2 items-start justify-between hover:border-zinc-700 transition-colors"):
                                ui.label(text).classes("text-zinc-200 text-sm flex-1 leading-relaxed")
                                ui.button("Kopieren", icon="content_copy", on_click=lambda t=text: ui.run_javascript(f"copyToClipboard({repr(t)});")).classes("btn-em shadow-none ml-4 flex-shrink-0").props("flat")

                    # Live Bottom: Tall Terminal
                    with ui.column().classes("w-full h-64 border-t border-zinc-900 bg-zinc-950 p-4 shrink-0 rounded-t-xl mx-4"):
                        ui.label("SYSTEM-LOG").classes("text-zinc-600 text-[10px] font-bold tracking-wider mb-2")
                        log_area = ui.log(max_lines=300).classes("w-full flex-1 bg-transparent text-zinc-500 font-mono text-xs p-0 border-none leading-relaxed")
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
                with ui.column().classes("w-full max-w-5xl mx-auto px-6 py-6 gap-4"):
                    ui.label("Einstellungen & Engine Parameter").classes("text-lg font-semibold tracking-tight text-zinc-100 mb-2")
                    
                    # Tight 3-column grid
                    with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 w-full"):
                        def build_model():
                            sel = ui.select(options=MODEL_OPTIONS, value=cfg.get("MODEL_SIZE", "large-v3")).classes("w-full").props("dark outlined")
                            refs["model"] = {"sel": sel}
                        config_item("Model Auswahl", build_model)

                        def buildStyle():
                            sel = ui.select(options=STYLE_OPTIONS, value=cfg.get("TRANSCRIPTION_STYLE_PRESET", "standard")).classes("w-full").props("dark outlined")
                            refs["style"] = {"sel": sel}
                        config_item("KI-Stil & Formulierung", buildStyle)

                        def build_sil():
                            sl = ui.slider(min=1.0, max=5.0, step=0.1, value=cfg.get("POST_SPEECH_SILENCE_DURATION", 3.0)).props("color=emerald-500 dark snap label")
                            refs["post_silence"] = {"slider": sl}
                        config_item("Satz-Cutoff / Denkpause (s)", build_sil)

                        def build_vad():
                            t = ui.switch("Enable Silero VAD", value=cfg.get("VAD_ENABLED", True)).classes("text-sm text-zinc-300").props("color=emerald-500 dark")
                            refs["vad"] = {"toggle": t}
                        config_item("Voice Activity Detection", build_vad)

                        def build_beam():
                            sel = ui.select(options=[1,3,5,10], value=cfg.get("BEAM_SIZE", 5)).classes("w-full").props("dark outlined")
                            refs["beam"] = {"sel": sel}
                        config_item("Beam Size (Präzision)", build_beam)

                        def build_lang():
                            sel = ui.select(options=LANGUAGE_OPTIONS, value=cfg.get("LANGUAGE", "de")).classes("w-full").props("dark outlined")
                            refs["lang"] = {"sel": sel}
                        config_item("Erzwungene Sprache", build_lang)

                        def build_device():
                            sel = ui.select(options=DEVICE_OPTIONS, value=cfg.get("DEVICE", "cuda")).classes("w-full").props("dark outlined")
                            refs["device"] = {"sel": sel}
                        config_item("Hardware Device", build_device)

                        def build_compute():
                            sel = ui.select(options=COMPUTE_OPTIONS, value=cfg.get("COMPUTE_TYPE", "float16")).classes("w-full").props("dark outlined")
                            refs["compute"] = {"sel": sel}
                        config_item("Compute Datentyp", build_compute)

                        def build_gpu():
                            inp = ui.number(value=cfg.get("GPU_DEVICE_INDEX", 0), format="%.0f", min=0).classes("w-full").props("dark outlined")
                            refs["gpu_idx"] = {"input": inp}
                        config_item("GPU Index-ID", build_gpu)

                        def build_mic():
                            cur = str(cfg.get("INPUT_DEVICE_INDEX")) if cfg.get("INPUT_DEVICE_INDEX") is not None else "none"
                            sel = ui.select(options=audio_devices, value=cur if cur in audio_devices else "none").classes("w-full").props("dark outlined")
                            refs["input_device"] = {"sel": sel}
                        config_item("Input Mikrofon", build_mic)
                        
                        def build_silero():
                            sl = ui.slider(min=0.0, max=1.0, step=0.05, value=cfg.get("SILERO_SENSITIVITY", 0.4)).props("color=emerald-500 dark snap label")
                            refs["silero"] = {"slider": sl}
                        config_item("VAD Sensibilität", build_silero)
                        
                        def build_min():
                            sl = ui.slider(min=0.1, max=3.0, step=0.1, value=cfg.get("MIN_LENGTH_OF_RECORDING", 0.5)).props("color=emerald-500 dark snap label")
                            refs["min_len"] = {"slider": sl}
                        config_item("Min. Audio-Länge (s)", build_min)

                        def build_pre_buf():
                            sl = ui.slider(min=0.0, max=5.0, step=0.1, value=cfg.get("PRE_RECORDING_BUFFER_DURATION", 1.0)).props("color=emerald-500 dark snap label")
                            refs["pre_buf"] = {"slider": sl}
                        config_item("Vorlauf-Puffer (s)", build_pre_buf)

                        def build_cursor():
                            t = ui.switch("Text direkt tippen", value=cfg.get("TYPE_INTO_CURSOR", False)).classes("text-sm text-zinc-300").props("color=emerald-500 dark")
                            refs["cursor"] = {"toggle": t}
                        config_item("Input Simulation", build_cursor)

                    with ui.row().classes("w-full mt-4 gap-4"):
                        with ui.column().classes("flex-1 gap-1 p-3 bg-zinc-900/70 border border-zinc-800/60 rounded-md"):
                            ui.label("Eigene Fachbegriffe (Vokabular)").classes("text-zinc-500 text-[10px] font-bold uppercase tracking-wider")
                            vocab_str = ", ".join(cfg.get("CUSTOM_VOCABULARY", []))
                            area = ui.textarea(placeholder="Begriff1, Begriff2, ...", value=vocab_str).classes("w-full").props("dark outlined rows=3")
                            refs["vocab"] = {"area": area}

                        with ui.column().classes("flex-1 gap-1 p-3 bg-zinc-900/70 border border-zinc-800/60 rounded-md"):
                            ui.label("Custom System-Prompt").classes("text-zinc-500 text-[10px] font-bold uppercase tracking-wider")
                            area = ui.textarea(placeholder="Anweisungen zur Formatierung...", value=cfg.get("INITIAL_PROMPT_EXTRA", "")).classes("w-full").props("dark outlined rows=3")
                            refs["prompt"] = {"area": area}


    def update_status():
        r = process_manager.is_running()
        if r:
            status_dot.classes(remove="bg-zinc-600", add="bg-emerald-500")
            status_label.set_text("Active")
            status_label.classes(remove="text-zinc-500", add="text-emerald-500")
            refs["main_btn"].set_text("Mikrofon deaktivieren")
            refs["main_btn"].props(remove="icon=mic", add="icon=stop")
            refs["main_btn"].classes(remove="btn-cta", add="btn-cta btn-cta-stop")
        else:
            status_dot.classes(remove="bg-emerald-500", add="bg-zinc-600")
            status_label.set_text("Offline")
            status_label.classes(remove="text-emerald-500", add="text-zinc-500")
            refs["main_btn"].set_text("Mikrofon aktivieren")
            refs["main_btn"].props(remove="icon=stop", add="icon=mic")
            refs["main_btn"].classes(remove="btn-cta btn-cta-stop", add="btn-cta")

    update_status()

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Whisper AI", port=8080, dark=True, reload=False, show=False)
