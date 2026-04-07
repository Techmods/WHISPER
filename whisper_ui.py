# =============================================================================
# whisper_ui.py — NiceGUI Web-Konfigurationsoberfläche für Whisper
#
# Dark Mode, Card-basiert (slate-900 / slate-800 / sky-400 Akzent)
# Läuft auf http://localhost:8080
#
# Starten: python whisper_ui.py
# =============================================================================

import asyncio
from nicegui import ui, app
import config_rw
import process_manager

# ---------------------------------------------------------------------------
# Modell-Optionen
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "tiny":                                        "tiny — 75 MB, sehr schnell",
    "base":                                        "base — 150 MB, schnell",
    "small":                                       "small — 500 MB, gute Balance",
    "medium":                                      "medium — 1.5 GB, gut",
    "large-v2":                                    "large-v2 — 3 GB, sehr gut",
    "large-v3":                                    "large-v3 — 3 GB, beste Qualität",
    "deepdml/faster-whisper-large-v3-turbo-ct2":   "large-v3-turbo — 1.6 GB, 8× schneller (empfohlen)",
    "Primeline/whisper-large-v3-turbo-german":     "large-v3-turbo-DE — 1.6 GB, nur Deutsch",
    "primeline/whisper-large-v3-german":           "large-v3-german — 3 GB, beste DE-Qualität",
    "MR-Eder/faster-whisper-large-v3-turbo-german":"MR-Eder turbo-DE — 1.6 GB, Deutsch-finetuned",
}

COMPUTE_OPTIONS = {
    "float16": "float16 — empfohlen (Blackwell/RTX 5080)",
    "int8":    "int8 — NICHT für Blackwell (sm_120)",
}

LANGUAGE_OPTIONS = {
    "de":   "Deutsch",
    "en":   "Englisch",
    "fr":   "Französisch",
    "es":   "Spanisch",
    "it":   "Italienisch",
    "auto": "Automatisch erkennen",
}

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
            # Linker Bereich: Icon-Container
            with ui.element("div").classes(
                "flex items-center justify-center w-16 h-16 min-w-[4rem] "
                "bg-slate-700 rounded-xl shadow-inner"
            ):
                ui.icon(icon).classes("text-sky-400 text-4xl")

            # Rechter Bereich: Titel, Wert, Beschreibung, Widget
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

    # In-Memory State für Vocabulary, Expansions, Corrections
    vocab_list: list = list(cfg.get("CUSTOM_VOCABULARY", []))
    expansions: list = [
        {"key": k, "val": v}
        for k, v in cfg.get("KEYWORD_EXPANSIONS", {}).items()
    ]
    corrections: list = [
        {"von": k, "nach": v}
        for k, v in cfg.get("CORRECTIONS", {}).items()
    ]

    # -----------------------------------------------------------------------
    # Globales Styling
    # -----------------------------------------------------------------------
    ui.add_head_html("""
    <style>
      body { background-color: #0f172a !important; }
      .nicegui-content { background-color: #0f172a !important; }
      /* Scrollbar */
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track { background: #1e293b; }
      ::-webkit-scrollbar-thumb { background: #38bdf8; border-radius: 3px; }
    </style>
    """)

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

        # Status-Badge
        status_dot = ui.element("span").classes(
            "w-3 h-3 rounded-full bg-red-500 inline-block"
        )
        status_label = ui.label("Gestoppt").classes("text-slate-400 text-sm")

    def update_status():
        running = process_manager.is_running()
        if running:
            status_dot.classes(remove="bg-red-500", add="bg-green-400")
            status_label.set_text("Läuft")
            status_label.classes(remove="text-slate-400", add="text-green-400")
            start_btn.set_text("Stoppen")
            start_btn.classes(
                remove="bg-sky-500 hover:bg-sky-400",
                add="bg-red-600 hover:bg-red-500",
            )
        else:
            status_dot.classes(remove="bg-green-400", add="bg-red-500")
            status_label.set_text("Gestoppt")
            status_label.classes(remove="text-green-400", add="text-slate-400")
            start_btn.set_text("Starten")
            start_btn.classes(
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
        with ui.column().classes("flex-1 gap-4 w-full"):

            # --- Modell ---
            model_select_ref = {}

            def build_model():
                sel = ui.select(
                    options=MODEL_OPTIONS,
                    value=cfg.get("MODEL_SIZE", "large-v3"),
                ).classes(
                    "w-full bg-slate-700 text-white rounded-xl"
                ).props("dark outlined dense")
                model_select_ref["sel"] = sel

            config_card(
                icon="computer",
                title="Modell",
                subtitle=MODEL_OPTIONS.get(cfg.get("MODEL_SIZE", "large-v3"), cfg.get("MODEL_SIZE", "")),
                description="Bestimmt Genauigkeit, Geschwindigkeit und VRAM-Bedarf.",
                build_widget=build_model,
            )

            # --- Compute Type ---
            compute_ref = {}

            def build_compute():
                sel = ui.select(
                    options=COMPUTE_OPTIONS,
                    value=cfg.get("COMPUTE_TYPE", "float16"),
                ).classes(
                    "w-full bg-slate-700 text-white rounded-xl"
                ).props("dark outlined dense")

                def on_compute_change(e):
                    if e.value == "int8":
                        ui.notify(
                            "INT8 ist für Blackwell (sm_120) deaktiviert — bitte float16 verwenden!",
                            type="negative",
                            timeout=6000,
                        )
                sel.on("update:model-value", on_compute_change)
                compute_ref["sel"] = sel

            config_card(
                icon="memory",
                title="Compute Type",
                subtitle=cfg.get("COMPUTE_TYPE", "float16"),
                description="Für RTX 5080 (Blackwell sm_120): zwingend float16.",
                build_widget=build_compute,
            )

            # --- Sprache ---
            lang_ref = {}

            def build_language():
                sel = ui.select(
                    options=LANGUAGE_OPTIONS,
                    value=cfg.get("LANGUAGE", "de"),
                ).classes(
                    "w-full bg-slate-700 text-white rounded-xl"
                ).props("dark outlined dense")
                lang_ref["sel"] = sel

            config_card(
                icon="language",
                title="Sprache",
                subtitle=LANGUAGE_OPTIONS.get(cfg.get("LANGUAGE", "de"), ""),
                description="Sprache der Spracherkennung. 'Auto' ist langsamer.",
                build_widget=build_language,
            )

            # --- Beam Size ---
            beam_label_ref = {}
            beam_ref = {}

            def build_beam():
                with ui.row().classes("items-center gap-4 w-full"):
                    slider = ui.slider(min=1, max=10, value=cfg.get("BEAM_SIZE", 5)).classes(
                        "flex-1"
                    ).props("color=sky-4 dark")
                    lbl = ui.label(str(cfg.get("BEAM_SIZE", 5))).classes(
                        "text-sky-400 text-lg font-medium w-6 text-right"
                    )
                    slider.on("update:model-value", lambda e: lbl.set_text(str(int(e.args))))
                    beam_ref["slider"] = slider
                    beam_label_ref["lbl"] = lbl

            config_card(
                icon="tune",
                title="Beam Size",
                subtitle=str(cfg.get("BEAM_SIZE", 5)),
                description="Dekodierungsgenauigkeit. 1 = schnell, 10 = präzise.",
                build_widget=build_beam,
            )

            # --- Custom Vocabulary ---
            vocab_row_ref = {}

            def build_vocab():
                vocab_input = ui.input(placeholder="Begriff hinzufügen…").classes(
                    "bg-slate-700 text-white rounded-xl flex-1"
                ).props("dark outlined dense")

                with ui.row().classes("flex-wrap gap-2 w-full") as chip_row:
                    vocab_row_ref["row"] = chip_row
                    for term in vocab_list:
                        _add_vocab_chip(chip_row, term, vocab_list)

                def add_vocab():
                    term = vocab_input.value.strip()
                    if term and term not in vocab_list:
                        vocab_list.append(term)
                        _add_vocab_chip(vocab_row_ref["row"], term, vocab_list)
                    vocab_input.set_value("")

                with ui.row().classes("items-center gap-2 w-full"):
                    vocab_input.on("keydown.enter", lambda: add_vocab())
                    ui.button("Hinzufügen", on_click=add_vocab).classes(
                        "bg-sky-500 hover:bg-sky-400 text-white text-xs rounded-lg px-3 py-1"
                    ).props("flat dense")

            config_card(
                icon="label",
                title="Custom Vocabulary",
                subtitle=f"{len(vocab_list)} Begriffe",
                description="Fachbegriffe die Whisper bevorzugt erkennen soll.",
                build_widget=build_vocab,
            )

            # --- Keyword Expansions ---
            exp_table_ref = {}

            def build_expansions():
                columns = [
                    {"name": "key", "label": "Schlüsselwort", "field": "key", "align": "left"},
                    {"name": "val", "label": "Ersetzungstext", "field": "val", "align": "left"},
                    {"name": "del", "label": "", "field": "del", "align": "center"},
                ]
                table = ui.table(
                    columns=columns,
                    rows=expansions,
                    row_key="key",
                ).classes("w-full bg-slate-700 text-white rounded-xl text-sm").props("dark flat dense")
                exp_table_ref["table"] = table

                table.add_slot("body-cell-del", """
                    <q-td :props="props">
                        <q-btn flat round dense icon="delete" color="red-4"
                            @click="$parent.$emit('delete-row', props.row)" />
                    </q-td>
                """)
                table.on("delete-row", lambda e: _delete_table_row(expansions, e.args, "key", table))

                with ui.row().classes("gap-2 mt-2"):
                    k_in = ui.input(placeholder="Schlüssel").classes(
                        "bg-slate-700 text-white rounded-xl"
                    ).props("dark outlined dense")
                    v_in = ui.input(placeholder="Ersetzung").classes(
                        "bg-slate-700 text-white rounded-xl"
                    ).props("dark outlined dense")

                    def add_exp():
                        k = k_in.value.strip()
                        v = v_in.value.strip()
                        if k:
                            expansions.append({"key": k, "val": v})
                            table.rows = expansions
                            table.update()
                            k_in.set_value("")
                            v_in.set_value("")

                    ui.button("+", on_click=add_exp).classes(
                        "bg-sky-500 hover:bg-sky-400 text-white rounded-lg px-3"
                    ).props("flat dense")

            config_card(
                icon="swap_horiz",
                title="Keyword Expansionen",
                subtitle=f"{len(expansions)} Einträge",
                description="Erkannte Abkürzungen werden automatisch ausgeschrieben.",
                build_widget=build_expansions,
            )

            # --- Corrections ---
            cor_table_ref = {}

            def build_corrections():
                columns = [
                    {"name": "von",  "label": "Von (Regex)", "field": "von",  "align": "left"},
                    {"name": "nach", "label": "Nach",        "field": "nach", "align": "left"},
                    {"name": "del",  "label": "",            "field": "del",  "align": "center"},
                ]
                table = ui.table(
                    columns=columns,
                    rows=corrections,
                    row_key="von",
                ).classes("w-full bg-slate-700 text-white rounded-xl text-sm").props("dark flat dense")
                cor_table_ref["table"] = table

                table.add_slot("body-cell-del", """
                    <q-td :props="props">
                        <q-btn flat round dense icon="delete" color="red-4"
                            @click="$parent.$emit('delete-row', props.row)" />
                    </q-td>
                """)
                table.on("delete-row", lambda e: _delete_table_row(corrections, e.args, "von", table))

                with ui.row().classes("gap-2 mt-2"):
                    v_in = ui.input(placeholder=r"Von (z.B. \bFalsch\b)").classes(
                        "bg-slate-700 text-white rounded-xl"
                    ).props("dark outlined dense")
                    n_in = ui.input(placeholder="Nach").classes(
                        "bg-slate-700 text-white rounded-xl"
                    ).props("dark outlined dense")

                    def add_cor():
                        v = v_in.value.strip()
                        n = n_in.value.strip()
                        if v:
                            corrections.append({"von": v, "nach": n})
                            table.rows = corrections
                            table.update()
                            v_in.set_value("")
                            n_in.set_value("")

                    ui.button("+", on_click=add_cor).classes(
                        "bg-sky-500 hover:bg-sky-400 text-white rounded-lg px-3"
                    ).props("flat dense")

            config_card(
                icon="spellcheck",
                title="Korrekturen",
                subtitle=f"{len(corrections)} Regeln",
                description="Whisper-Erkennungsfehler automatisch korrigieren.",
                build_widget=build_corrections,
            )

            # --- Ausgabe-Einstellungen ---
            cursor_ref = {}
            output_file_ref = {}

            def build_output():
                with ui.column().classes("gap-3 w-full"):
                    with ui.row().classes("items-center gap-3"):
                        toggle = ui.switch(
                            text="Text an Cursor-Position tippen",
                            value=cfg.get("TYPE_INTO_CURSOR", True),
                        ).classes("text-white").props("color=sky-4 dark")
                        cursor_ref["toggle"] = toggle

                    out_input = ui.input(
                        label="Ausgabedatei (leer = deaktiviert)",
                        value=cfg.get("OUTPUT_FILE") or "",
                    ).classes(
                        "w-full bg-slate-700 text-white rounded-xl"
                    ).props("dark outlined dense")
                    output_file_ref["input"] = out_input

            config_card(
                icon="output",
                title="Ausgabe",
                subtitle="Cursor-Tippen + Datei",
                description="Wohin der transkribierte Text geschrieben wird.",
                build_widget=build_output,
            )

            # --- Speichern-Button ---
            def save_config():
                new_val = model_select_ref["sel"].value
                updates = {
                    "MODEL_SIZE":              new_val,
                    "COMPUTE_TYPE":            compute_ref["sel"].value,
                    "LANGUAGE":               lang_ref["sel"].value,
                    "BEAM_SIZE":               int(beam_ref["slider"].value),
                    "CUSTOM_VOCABULARY":       list(vocab_list),
                    "KEYWORD_EXPANSIONS":      {r["key"]: r["val"] for r in expansions},
                    "CORRECTIONS":             {r["von"]: r["nach"] for r in corrections},
                    "TYPE_INTO_CURSOR":        cursor_ref["toggle"].value,
                    "OUTPUT_FILE":             output_file_ref["input"].value or None,
                }
                try:
                    config_rw.write_config(updates)
                    ui.notify("Konfiguration gespeichert.", type="positive")
                    if process_manager.is_running():
                        ui.notify(
                            "Neustart erforderlich damit Änderungen wirksam werden.",
                            type="warning",
                        )
                except Exception as e:
                    ui.notify(f"Fehler beim Speichern: {e}", type="negative")

            ui.button("Speichern", on_click=save_config).classes(
                "w-full bg-sky-500 hover:bg-sky-400 text-white font-medium "
                "rounded-2xl py-3 mt-2 shadow-lg"
            ).props("flat")

        # ==================================================================
        # RECHTE SPALTE — Steuerung + Live-Log
        # ==================================================================
        with ui.column().classes("w-full lg:w-96 gap-4"):

            # --- Start/Stop ---
            with ui.card().classes(
                "w-full bg-slate-800 rounded-2xl shadow-xl p-5"
            ):
                with ui.row().classes("items-center gap-3 mb-4"):
                    ui.icon("play_circle").classes("text-sky-400 text-3xl")
                    ui.label("Steuerung").classes("text-white text-lg font-medium")

                with ui.row().classes("items-center gap-3 mb-4"):
                    status_dot  # bereits oben definiert
                    status_label

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

                # Status beim Laden setzen
                update_status()

            # --- Live-Log ---
            with ui.card().classes(
                "w-full bg-slate-800 rounded-2xl shadow-xl p-5"
            ):
                with ui.row().classes("items-center gap-3 mb-3"):
                    ui.icon("subtitles").classes("text-sky-400 text-3xl")
                    ui.label("Live-Transkription").classes("text-white text-lg font-medium")

                log = ui.log(max_lines=200).classes(
                    "w-full h-96 bg-slate-900 text-slate-300 text-sm "
                    "rounded-xl font-mono"
                )

                # Bereits im Puffer vorhandene Zeilen einfügen
                for line in process_manager.get_log_buffer():
                    log.push(line)

                # Neue Zeilen live nachschieben
                process_manager.on_new_line(lambda line: log.push(line))

                ui.button("Log leeren", on_click=log.clear).classes(
                    "mt-2 text-slate-400 text-xs"
                ).props("flat dense")


# ---------------------------------------------------------------------------
# Hilfsfunktionen für dynamische Widgets
# ---------------------------------------------------------------------------

def _add_vocab_chip(container, term: str, vocab_list: list):
    """Fügt einen löschbaren Chip für einen Vocabulary-Begriff hinzu."""
    with container:
        with ui.row().classes(
            "items-center gap-1 bg-slate-700 text-sky-400 text-xs "
            "rounded-full px-3 py-1"
        ):
            ui.label(term)
            ui.button(
                icon="close",
                on_click=lambda t=term: _remove_vocab(t, vocab_list, container),
            ).classes("text-slate-400 hover:text-red-400 w-4 h-4").props(
                "flat round dense size=xs"
            )


def _remove_vocab(term: str, vocab_list: list, container):
    """Entfernt einen Begriff aus der Vocabulary-Liste und re-rendert die Chips."""
    if term in vocab_list:
        vocab_list.remove(term)
    container.clear()
    for t in vocab_list:
        _add_vocab_chip(container, t, vocab_list)


def _delete_table_row(rows: list, row_data: dict, key: str, table):
    """Entfernt eine Zeile aus einer Tabelle anhand des Key-Felds."""
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
