# =============================================================================
# whisper_ui.py — NiceGUI Web-Konfigurationsoberfläche für Whisper
#
# UI/UX Shadcn Edition (Minimalistisch, Technical, Grid-Layout, Audio-Visualizer)
# Läuft auf http://localhost:8080
# =============================================================================

import asyncio
import tkinter as tk
import tkinter.filedialog as fd
from nicegui import ui, app
import sounddevice as sd
import config_rw
import process_manager

# ---------------------------------------------------------------------------
# Optionen / Konstanten
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "tiny": "tiny (~75 MB)",
    "base": "base (~150 MB)",
    "small": "small (~500 MB)",
    "medium": "medium (~1.5 GB)",
    "large-v2": "large-v2 (~3 GB)",
    "large-v3": "large-v3 (~3 GB)",
    "deepdml/faster-whisper-large-v3-turbo-ct2": "large-v3-turbo (~1.6 GB) [Empfohlen]",
    "Primeline/whisper-large-v3-turbo-german": "large-v3-turbo-DE (~1.6 GB)",
    "primeline/whisper-large-v3-german": "large-v3-german (~3 GB)",
    "MR-Eder/faster-whisper-large-v3-turbo-german": "MR-Eder turbo-DE",
}

COMPUTE_OPTIONS = {"float16": "float16", "bfloat16": "bfloat16", "int8_float16": "int8_float16", "int8": "int8"}
DEVICE_OPTIONS = {"cuda": "GPU (CUDA)", "cpu": "CPU"}
LANGUAGE_OPTIONS = {"de": "Deutsch", "en": "Englisch", "auto": "Automatisch"}

def get_audio_devices() -> dict:
    devices = {"none": "System-Standard"}
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices[str(i)] = f"{i}: {dev['name']}"
    except Exception:
        pass
    return devices

# ---------------------------------------------------------------------------
# UI Style Widgets
# ---------------------------------------------------------------------------
def config_item(title: str, build_widget):
    """Rendert ein kompaktes, minimalistisches Einstellungs-Feld."""
    with ui.column().classes("w-full gap-1.5 p-3 bg-zinc-900/50 border border-zinc-800 rounded-md transition-colors hover:bg-zinc-900 hover:border-zinc-700"):
        ui.label(title).classes("text-zinc-400 text-[0.65rem] font-semibold uppercase tracking-wider")
        with ui.element("div").classes("w-full"):
            build_widget()

# ---------------------------------------------------------------------------
# Haupt-UI
# ---------------------------------------------------------------------------
@ui.page("/")
async def index():
    cfg = config_rw.read_config()
    audio_devices = get_audio_devices()

    vocab_list: list = list(cfg.get("CUSTOM_VOCABULARY", []))
    expansions: list = [{"key": k, "val": v} for k, v in cfg.get("KEYWORD_EXPANSIONS", {}).items()]
    corrections: list = [{"von": k, "nach": v} for k, v in cfg.get("CORRECTIONS", {}).items()]
    refs = {}

    # -----------------------------------------------------------------------
    # Head & Javascript (Shadcn + Audio Visualizer)
    # -----------------------------------------------------------------------
    ui.add_head_html("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      body, .nicegui-content { background-color: #09090b !important; color: #fff; font-family: 'Inter', sans-serif; margin: 0; padding: 0 !important; max-width: 100vw; height: 100vh; overflow: hidden; }
      
      /* Tailored Scrollbar for dense lists */
      ::-webkit-scrollbar { width: 4px; height: 4px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 2px; }
      ::-webkit-scrollbar-thumb:hover { background: #52525b; }

      /* Shadcn Dense Tab Overrides */
      .q-tab { min-height: 36px !important; padding: 0 16px !important; border-radius: 6px !important; }
      .q-tabs { background: #09090b; padding: 4px; border: 1px solid #27272a; border-radius: 8px; }
      .q-tab--active { background: #27272a !important; color: #fff !important; }
      
      /* Inputs overrides */
      .q-field__control { border-radius: 6px !important; }
    </style>
    
    <script>
    // Audio Visualizer Script
    let audioCtx; let analyser; let dataArray; let canvasCtx; let animId;
    async function initAudioVisualizer() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 128;
            analyser.smoothingTimeConstant = 0.8;
            const source = audioCtx.createMediaStreamSource(stream);
            source.connect(analyser);
            
            const canvas = document.getElementById('audio-canvas');
            if(canvas) {
                canvasCtx = canvas.getContext('2d');
                dataArray = new Uint8Array(analyser.frequencyBinCount);
                drawWave();
            }
        } catch(err) { console.warn('Mic access denied completely normal if blocked:', err); }
    }
    function drawWave() {
        animId = requestAnimationFrame(drawWave);
        const canvas = document.getElementById('audio-canvas');
        if(!canvas || !canvasCtx) return;
        analyser.getByteFrequencyData(dataArray);
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        
        canvasCtx.fillStyle = '#10b981'; // Emerald 500 for live signal
        const barWidth = (canvas.width / dataArray.length) * 2;
        let x = 0;
        for(let i = 0; i < dataArray.length; i++) {
            const barHeight = dataArray[i] / 4;
            // Draw centered
            const y = (canvas.height - barHeight) / 2;
            canvasCtx.fillRect(x, y, barWidth, barHeight || 1);
            x += barWidth + 1;
        }
    }
    // Setup Canvas observer to attach context once the UI renders it
    setTimeout(() => { if(document.getElementById('audio-canvas')) initAudioVisualizer(); }, 1500);
    </script>
    """)

    def save_config():
        device_val = refs.get("input_device", {}).get("sel")
        device_index = None
        if device_val and device_val.value and device_val.value != "none":
            try:
                device_index = int(device_val.value)
            except ValueError:
                pass

        updates = {
            "MODEL_SIZE":                refs["model"]["sel"].value,
            "COMPUTE_TYPE":              refs["compute"]["sel"].value,
            "DEVICE":                    refs["device"]["sel"].value,
            "GPU_DEVICE_INDEX":          int(refs["gpu_idx"]["input"].value),
            "LANGUAGE":                  refs["lang"]["sel"].value,
            "BEAM_SIZE":                 int(refs["beam"]["sel"].value),
            "INITIAL_PROMPT_EXTRA":      refs["prompt"]["area"].value,
            "CUSTOM_VOCABULARY":         list(vocab_list),
            "KEYWORD_EXPANSIONS":        {r["key"]: r["val"] for r in expansions},
            "CORRECTIONS":               {r["von"]: r["nach"] for r in corrections},
            "INPUT_DEVICE_INDEX":        device_index,
            "VAD_ENABLED":               refs["vad"]["toggle"].value,
            "SILERO_SENSITIVITY":        round(refs["silero"]["slider"].value, 2),
            "MIN_LENGTH_OF_RECORDING":   round(refs["min_len"]["slider"].value, 1),
            "PRE_RECORDING_BUFFER_DURATION": round(refs["pre_buf"]["slider"].value, 1),
            "TYPE_INTO_CURSOR":          refs["cursor"]["toggle"].value,
            "OUTPUT_FILE":               refs["outfile"]["input"].value or None,
        }
        try:
            config_rw.write_config(updates)
            ui.notify("Config saved.", type="positive", color="emerald-600", position="bottom-right")
            if process_manager.is_running():
                ui.notify("Neustart erforderlich", type="warning", color="amber-600", position="bottom-right")
        except Exception as e:
            ui.notify(f"Error: {e}", type="negative", color="red-600", position="bottom-right")

    # -----------------------------------------------------------------------
    # Layout (App-Frame)
    # -----------------------------------------------------------------------
    with ui.column().classes("w-full h-full"):
        
        # --- HEADER ---
        with ui.row().classes("w-full h-14 items-center justify-between px-6 border-b border-zinc-800 bg-zinc-950"):
            with ui.row().classes("items-center gap-2"):
                ui.element("div").classes("w-2 h-2 rounded-full bg-zinc-100")
                ui.label("Whisper").classes("text-zinc-100 font-semibold tracking-tight")
            
            with ui.row().classes("items-center gap-4"):
                status_dot = ui.element("div").classes("w-2 h-2 rounded-full bg-zinc-600")
                status_label = ui.label("Idle").classes("text-zinc-400 text-xs font-medium uppercase")
                ui.button("Save", on_click=save_config).classes("bg-zinc-100 hover:bg-zinc-200 text-zinc-900 rounded-md px-4 py-1 text-xs font-semibold normal-case shadow-none").props("flat")

        # --- MAIN CONTENT GRID ---
        with ui.row().classes("w-full flex-1 overflow-hidden flex-nowrap"):
            
            # LEFT SIDE (SETTINGS) - Takes 60%
            with ui.column().classes("w-[60%] h-full border-r border-zinc-800 bg-zinc-950 flex flex-col"):
                
                # Tab Navigation
                with ui.row().classes("w-full px-6 py-4 border-b border-zinc-800/50"):
                    with ui.tabs().props("dense no-caps inline-label indicator-color=transparent") as tabs:
                        tab_hw = ui.tab('System')
                        tab_audio = ui.tab('Audio')
                        tab_post = ui.tab('Verarbeitung')

                # Tab Panels
                with ui.tab_panels(tabs, value=tab_hw).classes("w-full flex-1 bg-transparent p-6 overflow-y-auto"):
                    
                    # SYSTEM TAB
                    with ui.tab_panel(tab_hw).classes("p-0 w-full"):
                        with ui.element("div").classes("grid grid-cols-2 gap-4 w-full"):
                            def build_model():
                                sel = ui.select(options=MODEL_OPTIONS, value=cfg.get("MODEL_SIZE", "large-v3")).classes("w-full").props("dark outlined dense")
                                refs["model"] = {"sel": sel}
                            config_item("Model", build_model)

                            def build_lang():
                                sel = ui.select(options=LANGUAGE_OPTIONS, value=cfg.get("LANGUAGE", "de")).classes("w-full").props("dark outlined dense")
                                refs["lang"] = {"sel": sel}
                            config_item("Language", build_lang)

                            def build_device():
                                sel = ui.select(options=DEVICE_OPTIONS, value=cfg.get("DEVICE", "cuda")).classes("w-full").props("dark outlined dense")
                                refs["device"] = {"sel": sel}
                            config_item("Hardware Device", build_device)

                            def build_compute():
                                sel = ui.select(options=COMPUTE_OPTIONS, value=cfg.get("COMPUTE_TYPE", "float16")).classes("w-full").props("dark outlined dense")
                                refs["compute"] = {"sel": sel}
                            config_item("Compute Type", build_compute)

                            def build_gpu():
                                inp = ui.number(value=cfg.get("GPU_DEVICE_INDEX", 0), format="%.0f", min=0).classes("w-full").props("dark outlined dense")
                                refs["gpu_idx"] = {"input": inp}
                            config_item("GPU Index", build_gpu)

                            def build_beam():
                                sel = ui.select(options=[1,3,5,10], value=cfg.get("BEAM_SIZE", 5)).classes("w-full").props("dark outlined dense")
                                refs["beam"] = {"sel": sel}
                            config_item("Beam Size (Accuracy)", build_beam)

                    # AUDIO TAB
                    with ui.tab_panel(tab_audio).classes("p-0 w-full"):
                        with ui.element("div").classes("grid grid-cols-2 gap-4 w-full"):
                            def build_mic():
                                cur = str(cfg.get("INPUT_DEVICE_INDEX")) if cfg.get("INPUT_DEVICE_INDEX") is not None else "none"
                                sel = ui.select(options=audio_devices, value=cur if cur in audio_devices else "none").classes("w-full").props("dark outlined dense")
                                refs["input_device"] = {"sel": sel}
                            config_item("Input Device", build_mic)

                            def build_vad():
                                t = ui.switch("Enable Silero VAD", value=cfg.get("VAD_ENABLED", True)).classes("text-sm text-zinc-300").props("color=zinc-200 dark")
                                refs["vad"] = {"toggle": t}
                            config_item("Voice Activity Detection", build_vad)

                            def build_silero():
                                sl = ui.slider(min=0.0, max=1.0, step=0.05, value=cfg.get("SILERO_SENSITIVITY", 0.4)).props("color=zinc-400 dark snap")
                                refs["silero"] = {"slider": sl}
                            config_item("VAD Sensitivity", build_silero)

                            def build_min():
                                sl = ui.slider(min=0.1, max=3.0, step=0.1, value=cfg.get("MIN_LENGTH_OF_RECORDING", 0.5)).props("color=zinc-400 dark snap")
                                refs["min_len"] = {"slider": sl}
                            config_item("Min. Record Length (s)", build_min)

                            def build_pre():
                                sl = ui.slider(min=0.0, max=1.0, step=0.1, value=cfg.get("PRE_RECORDING_BUFFER_DURATION", 0.2)).props("color=zinc-400 dark snap")
                                refs["pre_buf"] = {"slider": sl}
                            config_item("Pre-Recording Buffer (s)", build_pre)

                    # VERARBEITUNG TAB (Includes Output for density)
                    with ui.tab_panel(tab_post).classes("p-0 w-full"):
                        with ui.element("div").classes("grid grid-cols-2 gap-4 w-full mb-4"):
                            def build_cursor():
                                t = ui.switch("Eingabe an Maus-Cursor", value=cfg.get("TYPE_INTO_CURSOR", True)).classes("text-sm text-zinc-300").props("color=zinc-200 dark")
                                refs["cursor"] = {"toggle": t}
                            config_item("Cursor Action", build_cursor)

                            def build_out():
                                with ui.row().classes("w-full items-center gap-2"):
                                    out_input = ui.input(placeholder="Pfad zu transkription.txt", value=cfg.get("OUTPUT_FILE") or "").classes("flex-1").props("dark outlined dense")
                                    ui.button(icon="folder", on_click=lambda: out_input.set_value(fd.asksaveasfilename(defaultextension=".txt")) if fd else None).classes("bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-md px-3 h-10 shadow-none").props("flat")
                                    refs["outfile"] = {"input": out_input}
                            config_item("Logfile Output", build_out)
                        
                        def build_prompt():
                            area = ui.textarea(placeholder="Kontext für das Modell...", value=cfg.get("INITIAL_PROMPT_EXTRA", "")).classes("w-full").props("dark outlined dense rows=2")
                            refs["prompt"] = {"area": area}
                        config_item("Initial Prompt", build_prompt)

                        vocab_row_ref = {}
                        def build_vocab():
                            with ui.row().classes("w-full items-center gap-2 mb-2"):
                                inp = ui.input(placeholder="Fachbegriff...").classes("flex-1").props("dark outlined dense")
                                def add_v():
                                    if inp.value.strip():
                                        vocab_list.append(inp.value.strip())
                                        _remove_vocab("", vocab_list, vocab_row_ref["row"])
                                        inp.set_value("")
                                inp.on("keydown.enter", add_v)
                                ui.button(icon="add", on_click=add_v).classes("bg-zinc-800 text-zinc-300 hover:bg-zinc-700 rounded-md shadow-none py-1 h-10").props("flat")
                            with ui.row().classes("flex-wrap gap-2 w-full") as crow:
                                vocab_row_ref["row"] = crow
                                for t in vocab_list: _add_vocab_chip(crow, t, vocab_list)
                        config_item("Vocabulary", build_vocab)

            # RIGHT SIDE (EXECUTION & LOGS) - Takes 40%
            with ui.column().classes("w-[40%] h-full bg-zinc-950 flex flex-col pt-0"):
                
                # Signal Visualizer (Top Segment)
                with ui.column().classes("w-full h-32 border-b border-zinc-800 flex items-center justify-center p-6 bg-[#0a0a0c]"):
                    ui.label("LIVE SIGNAL").classes("text-zinc-600 text-[0.65rem] font-bold tracking-widest absolute top-4 left-6")
                    ui.html('<canvas id="audio-canvas" width="200" height="30"></canvas>').classes("mx-auto mt-4 opacity-80")

                # Transcription Control (Middle)
                with ui.column().classes("w-full p-6 border-b border-zinc-800 bg-zinc-900/20"):
                    async def toggle_transcription():
                        if process_manager.is_running():
                            await process_manager.stop_transcription()
                        else:
                            await process_manager.start_transcription()
                        update_status()

                    start_btn = ui.button("Listen", on_click=toggle_transcription).classes(
                        "w-full bg-zinc-100 hover:bg-white text-zinc-900 font-semibold text-sm "
                        "rounded-md py-3 transition-colors shadow-sm normal-case"
                    ).props("flat")
                    refs["start_btn"] = start_btn

                # Terminal (Bottom)
                with ui.column().classes("w-full flex-1 p-0 relative"):
                    ui.label("TERMINAL").classes("text-zinc-600 text-[0.65rem] font-bold tracking-widest absolute top-4 left-6 z-10")
                    log_area = ui.log(max_lines=200).classes("w-full h-full bg-transparent text-emerald-500 font-mono text-xs pt-12 p-6 border-none")
                    refs["log"] = log_area
                    for line in process_manager.get_log_buffer(): log_area.push(line)
                    process_manager.on_new_line(lambda line: log_area.push(line))

    # -----------------------------------------------------------------------
    # Status Updater
    # -----------------------------------------------------------------------
    def update_status():
        r = process_manager.is_running()
        if r:
            status_dot.classes(remove="bg-zinc-600", add="bg-emerald-500")
            status_label.set_text("Active")
            status_label.classes(remove="text-zinc-400", add="text-emerald-500")
            refs["start_btn"].set_text("Stop Listening")
            refs["start_btn"].classes(remove="bg-zinc-100 hover:bg-white text-zinc-900", add="bg-rose-900/30 hover:bg-rose-900/50 text-rose-400 border border-rose-900/50")
        else:
            status_dot.classes(remove="bg-emerald-500", add="bg-zinc-600")
            status_label.set_text("Idle")
            status_label.classes(remove="text-emerald-500", add="text-zinc-400")
            refs["start_btn"].set_text("Listen")
            refs["start_btn"].classes(remove="bg-rose-900/30 hover:bg-rose-900/50 text-rose-400 border border-rose-900/50", add="bg-zinc-100 hover:bg-white text-zinc-900")

    update_status()

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _add_vocab_chip(container, term: str, vocab_list: list):
    with container:
        with ui.row().classes("items-center gap-1.5 border border-zinc-700 bg-zinc-800 text-zinc-300 text-xs rounded-sm pl-2 pr-1 py-0.5"):
            ui.label(term)
            ui.button(icon="close", on_click=lambda t=term: _remove_vocab(t, vocab_list, container)).classes("text-zinc-500 hover:text-rose-400 w-4 h-4").props("flat round dense size=xs")

def _remove_vocab(term, vlist, container):
    if term in vlist: vlist.remove(term)
    container.clear()
    for t in vlist: _add_vocab_chip(container, t, vlist)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Whisper Settings", port=8080, dark=True, reload=False, show=False)
