# =============================================================================
# whisper_ui.py — NiceGUI Web-Konfigurationsoberfläche für Whisper
#
# Dark Mode, Card-basiert (slate-900 / slate-800 / sky-400 Akzent)
# Läuft auf http://localhost:8080
#
# Starten: python whisper_ui.py
# =============================================================================

import asyncio
import tkinter as tk
import tkinter.filedialog as fd
from nicegui import ui, app
import sounddevice as sd
import config_rw
import process_manager

# ---------------------------------------------------------------------------
# Modell-Optionen
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "tiny":                                         "tiny — 75 MB, sehr schnell",
    "base":                                         "base — 150 MB, schnell",
    "small":                                        "small — 500 MB, gute Balance",
    "medium":                                       "medium — 1.5 GB, gut",
    "large-v2":                                     "large-v2 — 3 GB, sehr gut",
    "large-v3":                                     "large-v3 — 3 GB, beste Qualität",
    "deepdml/faster-whisper-large-v3-turbo-ct2":    "large-v3-turbo — 1.6 GB, 8× schneller (empfohlen)",
    "Primeline/whisper-large-v3-turbo-german":      "large-v3-turbo-DE — 1.6 GB, nur Deutsch",
    "primeline/whisper-large-v3-german":            "large-v3-german — 3 GB, beste DE-Qualität",
    "MR-Eder/faster-whisper-large-v3-turbo-german": "MR-Eder turbo-DE — Deutsch-finetuned",
}

COMPUTE_OPTIONS = {
    "float16":      "float16 — empfohlen (Blackwell/RTX 5080)",
    "bfloat16":     "bfloat16 — Alternative zu float16",
    "int8_float16": "int8_float16 — weniger VRAM (nicht für Blackwell)",
    "int8":         "int8 — NICHT für Blackwell (sm_120)",
}

LANGUAGE_OPTIONS = {
    "de":   "Deutsch",
    "en":   "Englisch",
    "fr":   "Französisch",
    "es":   "Spanisch",
    "it":   "Italienisch",
    "auto": "Automatisch erkennen",
}


def get_audio_devices() -> dict:
    """Gibt verfügbare Eingabegeräte als {index_str: name} zurück."""
    devices = {"none": "System-Standard"}
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices[str(i)] = f"{i}: {dev['name']}"
    except Exception:
        pass
    return devices


# ---------------------------------------------------------------------------
# Hilfsfunktion: Konfig-Card mit Icon
# ---------------------------------------------------------------------------
def config_card(icon: str, title: str, subtitle: str, description: str, build_widget):
    """Rendert eine Einstellungs-Karte im TailwindCSS-Album-Stil."""
    with ui.card().classes(
        "w-full bg-slate-800 rounded-2xl shadow-xl p-0 overflow-hidden"
    ):
        with ui.row().classes(
            "flex-col md:flex-row items-center md:items-start gap-4 p-5"
        ):
            with ui.element("div").classes(
                "flex items-center justify-center w-16 h-16 min-w-[4rem] "
                "bg-slate-700 rounded-xl shadow-inner"
            ):
                ui.icon(icon).classes("text-sky-400 text-4xl")

            with ui.column().classes("gap-1 flex-1 w-full"):
                ui.label(title).classes("text-white text-lg font-medium leading-tight")
                ui.label(subtitle).classes("text-sky-400 text-sm font-medium")
                ui.label(description).classes("text-slate-400 text-xs mb-2")
                build_widget()


# ---------------------------------------------------------------------------
# Haupt-UI
# ---------------------------------------------------------------------------
@ui.page("/")
async def index():
    cfg = config_rw.read_config()
    audio_devices = get_audio_devices()

    # In-Memory State
    vocab_list: list = list(cfg.get("CUSTOM_VOCABULARY", []))
    expansions: list = [
        {"key": k, "val": v}
        for k, v in cfg.get("KEYWORD_EXPANSIONS", {}).items()
    ]
    corrections: list = [
        {"von": k, "nach": v}
        for k, v in cfg.get("CORRECTIONS", {}).items()
    ]

    # Widget-Referenzen
    refs = {}

    # -----------------------------------------------------------------------
    # Globales Styling
    # -----------------------------------------------------------------------
    ui.add_head_html("""
    <style>
      body, .nicegui-content { background-color: #0f172a !important; }
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track { background: #1e293b; }
      ::-webkit-scrollbar-thumb { background: #38bdf8; border-radius: 3px; }
    </style>
    """)

    # -----------------------------------------------------------------------
    # Collect + Save (zentral, wird von Header-Button und manuell aufgerufen)
    # -----------------------------------------------------------------------
    def save_config():
        device_val = refs.get("input_device", {}).get("sel")
        device_index = None
        if device_val and device_val.value and device_val.value != "none":
            try:
                device_index = int(device_val.value)
            except ValueError:
                device_index = None

        updates = {
            "MODEL_SIZE":                refs["model"]["sel"].value,
            "COMPUTE_TYPE":              refs["compute"]["sel"].value,
            "LANGUAGE":                  refs["lang"]["sel"].value,
            "BEAM_SIZE":                 int(refs["beam"]["slider"].value),
            "INITIAL_PROMPT_EXTRA":      refs["prompt"]["area"].value,
            "CUSTOM_VOCABULARY":         list(vocab_list),
            "KEYWORD_EXPANSIONS":        {r["key"]: r["val"] for r in expansions},
            "CORRECTIONS":               {r["von"]: r["nach"] for r in corrections},
            "INPUT_DEVICE_INDEX":        device_index,
            "VAD_ENABLED":               refs["vad"]["toggle"].value,
            "SILERO_SENSITIVITY":        round(refs["silero"]["slider"].value, 2),
            "TYPE_INTO_CURSOR":          refs["cursor"]["toggle"].value,
            "OUTPUT_FILE":               refs["outfile"]["input"].value or None,
        }
        try:
            config_rw.write_config(updates)
            ui.notify("Konfiguration gespeichert.", type="positive")
            if process_manager.is_running():
                ui.notify("Neustart erforderlich damit Änderungen wirksam werden.", type="warning")
        except Exception as e:
            ui.notify(f"Fehler beim Speichern: {e}", type="negative")

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    with ui.row().classes(
        "w-full items-center justify-between px-6 py-4 "
        "bg-slate-800 rounded-2xl shadow-xl mb-6"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.icon("mic").classes("text-sky-400 text-3xl")
            ui.label("Whisper UI").classes("text-white text-2xl font-medium")

        with ui.row().classes("items-center gap-4"):
            # Status — Punkt direkt vor Text
            status_dot = ui.icon("circle").classes("text-red-500 text-sm")
            status_label = ui.label("Gestoppt").classes("text-slate-400 text-sm")

            ui.button("Speichern", icon="save", on_click=save_config).classes(
                "bg-sky-500 hover:bg-sky-400 text-white text-sm rounded-xl px-4"
            ).props("flat dense")

    def update_status():
        running = process_manager.is_running()
        if running:
            status_dot.classes(remove="text-red-500", add="text-green-400")
            status_label.set_text("Läuft")
            status_label.classes(remove="text-slate-400", add="text-green-400")
            refs["start_btn"].set_text("Stoppen")
            refs["start_btn"].classes(
                remove="bg-sky-500 hover:bg-sky-400",
                add="bg-red-600 hover:bg-red-500",
            )
        else:
            status_dot.classes(remove="text-green-400", add="text-red-500")
            status_label.set_text("Gestoppt")
            status_label.classes(remove="text-green-400", add="text-slate-400")
            refs["start_btn"].set_text("Starten")
            refs["start_btn"].classes(
                remove="bg-red-600 hover:bg-red-500",
                add="bg-sky-500 hover:bg-sky-400",
            )

    # -----------------------------------------------------------------------
    # 2-Spalten-Layout
    # -----------------------------------------------------------------------
    with ui.row().classes("w-full gap-6 items-start flex-col lg:flex-row"):

        # ==================================================================
        # LINKE SPALTE — Konfiguration
        # ==================================================================
        with ui.column().classes("flex-1 gap-4 w-full min-w-0"):

            # --- Modell ---
            def build_model():
                sel = ui.select(
                    options=MODEL_OPTIONS,
                    value=cfg.get("MODEL_SIZE", "large-v3"),
                ).classes("w-full bg-slate-700 text-white rounded-xl").props("dark outlined dense")
                refs["model"] = {"sel": sel}

            config_card("computer", "Modell",
                MODEL_OPTIONS.get(cfg.get("MODEL_SIZE", "large-v3"), cfg.get("MODEL_SIZE", "")),
                "Bestimmt Genauigkeit, Geschwindigkeit und VRAM-Bedarf.",
                build_model)

            # --- Compute Type ---
            def build_compute():
                sel = ui.select(
                    options=COMPUTE_OPTIONS,
                    value=cfg.get("COMPUTE_TYPE", "float16"),
                ).classes("w-full bg-slate-700 text-white rounded-xl").props("dark outlined dense")

                def on_compute_change(e):
                    if e.value in ("int8", "int8_float16"):
                        ui.notify(
                            "INT8/int8_float16 sind für Blackwell (sm_120) deaktiviert — bitte float16 oder bfloat16 verwenden!",
                            type="negative", timeout=6000,
                        )
                sel.on("update:model-value", on_compute_change)
                refs["compute"] = {"sel": sel}

            config_card("memory", "Compute Type",
                cfg.get("COMPUTE_TYPE", "float16"),
                "Für RTX 5080 (Blackwell sm_120): float16 oder bfloat16.",
                build_compute)

            # --- Sprache ---
            def build_language():
                sel = ui.select(
                    options=LANGUAGE_OPTIONS,
                    value=cfg.get("LANGUAGE", "de"),
                ).classes("w-full bg-slate-700 text-white rounded-xl").props("dark outlined dense")
                refs["lang"] = {"sel": sel}

            config_card("language", "Sprache",
                LANGUAGE_OPTIONS.get(cfg.get("LANGUAGE", "de"), ""),
                "Sprache der Spracherkennung. 'Auto' ist langsamer.",
                build_language)

            # --- Beam Size ---
            def build_beam():
                with ui.row().classes("items-center gap-4 w-full"):
                    slider = ui.slider(min=1, max=10, value=cfg.get("BEAM_SIZE", 5)).classes("flex-1").props("color=sky-4 dark")
                    lbl = ui.label(str(cfg.get("BEAM_SIZE", 5))).classes("text-sky-400 text-lg font-medium w-6 text-right")
                    slider.on("update:model-value", lambda e: lbl.set_text(str(int(e.args))))
                    refs["beam"] = {"slider": slider}

            config_card("tune", "Beam Size",
                str(cfg.get("BEAM_SIZE", 5)),
                "Dekodierungsgenauigkeit. 1 = schnell, 10 = präzise.",
                build_beam)

            # --- Initial Prompt ---
            def build_prompt():
                area = ui.textarea(
                    placeholder="Kontext für Whisper (Interpunktion, Stil, Formatierung)…",
                    value=cfg.get("INITIAL_PROMPT_EXTRA", ""),
                ).classes("w-full bg-slate-700 text-white rounded-xl").props("dark outlined dense rows=3")
                refs["prompt"] = {"area": area}

            config_card("edit_note", "Initial Prompt",
                "Kontext & Formatierungshinweise",
                "Gibt Whisper Stil-Vorgaben. Fachbegriffe werden automatisch aus dem Vokabular angehängt.",
                build_prompt)

            # --- Audio-Eingabequelle ---
            def build_audio_device():
                current_idx = cfg.get("INPUT_DEVICE_INDEX")
                current_val = str(current_idx) if current_idx is not None else "none"
                sel = ui.select(
                    options=audio_devices,
                    value=current_val if current_val in audio_devices else "none",
                ).classes("w-full bg-slate-700 text-white rounded-xl").props("dark outlined dense")
                refs["input_device"] = {"sel": sel}

            config_card("mic", "Audio-Eingabequelle",
                audio_devices.get(
                    str(cfg.get("INPUT_DEVICE_INDEX")) if cfg.get("INPUT_DEVICE_INDEX") is not None else "none",
                    "System-Standard"
                ),
                "Mikrofon oder virtuelles Kabel. System-Standard = Windows-Standardgerät.",
                build_audio_device)

            # --- VAD + Silero-Sensitivity ---
            def build_vad():
                with ui.column().classes("gap-3 w-full"):
                    toggle = ui.switch(
                        text="Voice Activity Detection (Silero VAD)",
                        value=cfg.get("VAD_ENABLED", True),
                    ).classes("text-white").props("color=sky-4 dark")
                    refs["vad"] = {"toggle": toggle}

                    with ui.row().classes("items-center gap-4 w-full"):
                        ui.label("Silero-Empfindlichkeit:").classes("text-slate-400 text-xs w-40")
                        slider = ui.slider(min=0.0, max=1.0, step=0.05, value=cfg.get("SILERO_SENSITIVITY", 0.4)).classes("flex-1").props("color=sky-4 dark")
                        lbl = ui.label(f"{cfg.get('SILERO_SENSITIVITY', 0.4):.2f}").classes("text-sky-400 text-sm w-10 text-right")
                        slider.on("update:model-value", lambda e: lbl.set_text(f"{float(e.args):.2f}"))
                        refs["silero"] = {"slider": slider}

            config_card("graphic_eq", "VAD & Empfindlichkeit",
                "Silero VAD",
                "Filtert Stille heraus, spart Rechenleistung und reduziert Halluzinationen.",
                build_vad)

            # --- Custom Vocabulary ---
            vocab_row_ref = {}

            def build_vocab():
                with ui.row().classes("items-center gap-2 w-full mb-2"):
                    vocab_input = ui.input(placeholder="Begriff hinzufügen…").classes(
                        "bg-slate-700 text-white rounded-xl flex-1"
                    ).props("dark outlined dense")

                    def add_vocab():
                        term = vocab_input.value.strip()
                        if term and term not in vocab_list:
                            vocab_list.append(term)
                            _add_vocab_chip(vocab_row_ref["row"], term, vocab_list)
                        vocab_input.set_value("")

                    vocab_input.on("keydown.enter", lambda: add_vocab())
                    ui.button(icon="add", on_click=add_vocab).classes(
                        "bg-sky-500 hover:bg-sky-400 text-white rounded-xl"
                    ).props("flat dense")

                with ui.row().classes("flex-wrap gap-2 w-full") as chip_row:
                    vocab_row_ref["row"] = chip_row
                    for term in vocab_list:
                        _add_vocab_chip(chip_row, term, vocab_list)

            config_card("label", "Custom Vocabulary",
                f"{len(vocab_list)} Begriffe",
                "Fachbegriffe die Whisper bevorzugt erkennen soll.",
                build_vocab)

            # --- Keyword Expansions ---
            def build_expansions():
                columns = [
                    {"name": "key", "label": "Schlüsselwort", "field": "key", "align": "left"},
                    {"name": "val", "label": "Ersetzungstext", "field": "val", "align": "left"},
                    {"name": "del", "label": "", "field": "del", "align": "center", "style": "width:40px"},
                ]
                table = ui.table(columns=columns, rows=expansions, row_key="key").classes(
                    "w-full bg-slate-700 text-white rounded-xl text-sm"
                ).props("dark flat dense")
                table.add_slot("body-cell-del", """
                    <q-td :props="props">
                        <q-btn flat round dense icon="delete" color="red-4"
                            @click="$parent.$emit('delete-row', props.row)" />
                    </q-td>
                """)
                table.on("delete-row", lambda e: _delete_table_row(expansions, e.args, "key", table))

                with ui.row().classes("gap-2 mt-2 items-center"):
                    k_in = ui.input(placeholder="Schlüssel").classes("bg-slate-700 text-white rounded-xl flex-1").props("dark outlined dense")
                    v_in = ui.input(placeholder="Ersetzung").classes("bg-slate-700 text-white rounded-xl flex-1").props("dark outlined dense")

                    def add_exp():
                        k, v = k_in.value.strip(), v_in.value.strip()
                        if k:
                            expansions.append({"key": k, "val": v})
                            table.rows = expansions
                            table.update()
                            k_in.set_value("")
                            v_in.set_value("")

                    ui.button(icon="add", on_click=add_exp).classes(
                        "bg-sky-500 hover:bg-sky-400 text-white rounded-xl"
                    ).props("flat dense")

            config_card("swap_horiz", "Keyword Expansionen",
                f"{len(expansions)} Einträge",
                "Erkannte Abkürzungen werden automatisch ausgeschrieben.",
                build_expansions)

            # --- Corrections ---
            def build_corrections():
                columns = [
                    {"name": "von",  "label": "Von (Regex)", "field": "von",  "align": "left"},
                    {"name": "nach", "label": "Nach",        "field": "nach", "align": "left"},
                    {"name": "del",  "label": "",            "field": "del",  "align": "center", "style": "width:40px"},
                ]
                table = ui.table(columns=columns, rows=corrections, row_key="von").classes(
                    "w-full bg-slate-700 text-white rounded-xl text-sm"
                ).props("dark flat dense")
                table.add_slot("body-cell-del", """
                    <q-td :props="props">
                        <q-btn flat round dense icon="delete" color="red-4"
                            @click="$parent.$emit('delete-row', props.row)" />
                    </q-td>
                """)
                table.on("delete-row", lambda e: _delete_table_row(corrections, e.args, "von", table))

                with ui.row().classes("gap-2 mt-2 items-center"):
                    v_in = ui.input(placeholder=r"Von (z.B. \bFalsch\b)").classes("bg-slate-700 text-white rounded-xl flex-1").props("dark outlined dense")
                    n_in = ui.input(placeholder="Nach").classes("bg-slate-700 text-white rounded-xl flex-1").props("dark outlined dense")

                    def add_cor():
                        v, n = v_in.value.strip(), n_in.value.strip()
                        if v:
                            corrections.append({"von": v, "nach": n})
                            table.rows = corrections
                            table.update()
                            v_in.set_value("")
                            n_in.set_value("")

                    ui.button(icon="add", on_click=add_cor).classes(
                        "bg-sky-500 hover:bg-sky-400 text-white rounded-xl"
                    ).props("flat dense")

            config_card("spellcheck", "Korrekturen",
                f"{len(corrections)} Regeln",
                "Whisper-Erkennungsfehler automatisch korrigieren.",
                build_corrections)

            # --- Ausgabe ---
            def build_output():
                with ui.column().classes("gap-3 w-full"):
                    toggle = ui.switch(
                        text="Text an Cursor-Position tippen",
                        value=cfg.get("TYPE_INTO_CURSOR", True),
                    ).classes("text-white").props("color=sky-4 dark")
                    refs["cursor"] = {"toggle": toggle}

                    with ui.row().classes("items-center gap-2 w-full"):
                        out_input = ui.input(
                            label="Ausgabedatei (leer = deaktiviert)",
                            value=cfg.get("OUTPUT_FILE") or "",
                        ).classes("bg-slate-700 text-white rounded-xl flex-1").props("dark outlined dense")
                        refs["outfile"] = {"input": out_input}

                        def pick_file():
                            root = tk.Tk()
                            root.withdraw()
                            root.wm_attributes("-topmost", True)
                            path = fd.asksaveasfilename(
                                title="Ausgabedatei wählen",
                                defaultextension=".txt",
                                filetypes=[("Textdatei", "*.txt"), ("Alle Dateien", "*.*")],
                                initialfile="transkription.txt",
                            )
                            root.destroy()
                            if path:
                                out_input.set_value(path)

                        ui.button(icon="folder_open", on_click=pick_file).classes(
                            "bg-slate-700 hover:bg-slate-600 text-sky-400 rounded-xl"
                        ).props("flat dense").tooltip("Datei auswählen")

            config_card("output", "Ausgabe",
                "Cursor-Tippen + Datei",
                "Wohin der transkribierte Text geschrieben wird.",
                build_output)

        # ==================================================================
        # RECHTE SPALTE — sticky, Steuerung + Live-Log
        # ==================================================================
        with ui.column().classes("w-full lg:w-96 gap-4").style("position: sticky; top: 1rem; align-self: flex-start;"):

            # --- Steuerung ---
            with ui.card().classes("w-full bg-slate-800 rounded-2xl shadow-xl p-5"):
                with ui.row().classes("items-center gap-3 mb-4"):
                    ui.icon("play_circle").classes("text-sky-400 text-3xl")
                    ui.label("Steuerung").classes("text-white text-lg font-medium")

                async def toggle_transcription():
                    if process_manager.is_running():
                        await process_manager.stop_transcription()
                    else:
                        await process_manager.start_transcription()
                    update_status()

                start_btn = ui.button("Starten", on_click=toggle_transcription).classes(
                    "w-full bg-sky-500 hover:bg-sky-400 text-white font-medium "
                    "rounded-xl py-2 shadow-lg"
                ).props("flat")
                refs["start_btn"] = start_btn
                update_status()

            # --- Live-Log ---
            with ui.card().classes("w-full bg-slate-800 rounded-2xl shadow-xl p-5"):
                with ui.row().classes("items-center gap-3 mb-3"):
                    ui.icon("subtitles").classes("text-sky-400 text-3xl")
                    ui.label("Live-Transkription").classes("text-white text-lg font-medium")

                log = ui.log(max_lines=200).classes(
                    "w-full h-96 bg-slate-900 text-slate-300 text-sm rounded-xl font-mono"
                )

                for line in process_manager.get_log_buffer():
                    log.push(line)

                process_manager.on_new_line(lambda line: log.push(line))

                ui.button("Log leeren", on_click=log.clear).classes(
                    "mt-2 text-slate-400 text-xs"
                ).props("flat dense")


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _add_vocab_chip(container, term: str, vocab_list: list):
    with container:
        with ui.row().classes(
            "items-center gap-1 bg-slate-700 text-sky-400 text-xs rounded-full px-3 py-1"
        ):
            ui.label(term)
            ui.button(
                icon="close",
                on_click=lambda t=term: _remove_vocab(t, vocab_list, container),
            ).classes("text-slate-400 hover:text-red-400 w-4 h-4").props("flat round dense size=xs")


def _remove_vocab(term: str, vocab_list: list, container):
    if term in vocab_list:
        vocab_list.remove(term)
    container.clear()
    for t in vocab_list:
        _add_vocab_chip(container, t, vocab_list)


def _delete_table_row(rows: list, row_data: dict, key: str, table):
    key_val = row_data.get(key)
    for i, r in enumerate(rows):
        if r.get(key) == key_val:
            rows.pop(i)
            break
    table.rows = rows
    table.update()


# ---------------------------------------------------------------------------
# App starten
# ---------------------------------------------------------------------------
ui.run(
    title="Whisper UI",
    port=8080,
    dark=True,
    favicon="🎙",
    reload=False,
)
