# =============================================================================
# transcription_system.py — Live-Transkriptionssystem
#
# Architektur:
#   - RealtimeSTT (AudioToTextRecorder) fuer Mikrofoneingabe und VAD
#   - faster-whisper (large-v3, float16, CUDA) fuer Spracherkennung
#   - Post-Processing: Korrekturen + Keyword-Expansionen (aus config.py)
#   - Ausgabe in Konsole mit Zeitstempel
#
# Voraussetzungen:
#   - RTX 5080 (Blackwell), CUDA 12.8, cuDNN 9
#   - CTranslate2 >= 4.6.3 (INT8 fuer sm_120 deaktiviert)
#   - compute_type="float16" zwingend (kein INT8 auf Blackwell)
#
# Starten:
#   python transcription_system.py
#
# Beenden: Strg+C
# =============================================================================

import re
import sys
import signal
import datetime
import time
from pathlib import Path

import pyautogui
import pyperclip

# Konfiguration aus separater Datei importieren
from config import (
    MODEL_SIZE,
    COMPUTE_TYPE,
    DEVICE,
    GPU_DEVICE_INDEX,
    LANGUAGE,
    BEAM_SIZE,
    CUSTOM_VOCABULARY,
    KEYWORD_EXPANSIONS,
    CORRECTIONS,
    INITIAL_PROMPT_EXTRA,
    SILERO_SENSITIVITY,
    MIN_LENGTH_OF_RECORDING,
    PRE_RECORDING_BUFFER_DURATION,
    TIMESTAMP_FORMAT,
    OUTPUT_SEPARATOR,
    TYPE_INTO_CURSOR,
    OUTPUT_FILE,
)

# pyautogui: kein Failsafe (Maus in Ecke wuerde sonst abbrechen)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Ausgabedatei oeffnen (append-Modus, bleibt ueber die gesamte Session offen)
_output_file = open(OUTPUT_FILE, "a", encoding="utf-8") if OUTPUT_FILE else None


# =============================================================================
# POST-PROCESSING
# =============================================================================

def apply_corrections(text: str) -> str:
    """
    Wendet regelbasierte Korrekturen auf den transkribierten Text an.

    Jeder Eintrag in CORRECTIONS ist ein regulaerer Ausdruck (Schluessel)
    der durch den zugehoerigen Ersetzungstext (Wert) ersetzt wird.
    Matching ist case-insensitive.

    Args:
        text: Roher transkribierter Text von Whisper.

    Returns:
        Text mit angewandten Korrekturen.
    """
    if not text:
        return text

    for pattern, replacement in CORRECTIONS.items():
        try:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        except re.error as e:
            # Ungueltige Regex nicht abwuergen, nur warnen
            print(f"  [WARNUNG] Ungueltige Regex in CORRECTIONS: '{pattern}' — {e}")

    return text


def apply_keyword_expansions(text: str) -> str:
    """
    Ersetzt Schluesselwoerter durch erweiterten Text (Keyword-Expansion).

    Wenn ein Schluessel aus KEYWORD_EXPANSIONS (Wortgrenze, case-insensitive)
    im Text vorkommt, wird er durch den zugehoerigen Expansionstext ersetzt.
    Laengere Schluessel werden zuerst geprueft (Reihenfolge in config.py beachten).

    Args:
        text: Text nach Korrekturen.

    Returns:
        Text mit angewandten Keyword-Expansionen.
    """
    if not text or not KEYWORD_EXPANSIONS:
        return text

    for keyword, expansion in KEYWORD_EXPANSIONS.items():
        # Wortgrenzen (\b) verhindern Teilwort-Ersetzungen
        pattern = r"\b" + re.escape(keyword) + r"\b"
        try:
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        except re.error as e:
            print(f"  [WARNUNG] Keyword-Expansion Fehler fuer '{keyword}': {e}")

    return text


def process_text(raw_text: str) -> str:
    """
    Vollstaendige Post-Processing-Pipeline fuer einen transkribierten Text.

    Reihenfolge:
        1. Korrekturen (Tipp-/Erkennungsfehler beheben)
        2. Keyword-Expansionen (Abkuerzungen ausschreiben)

    Args:
        raw_text: Direkte Ausgabe von Whisper/RealtimeSTT.

    Returns:
        Bereinigter und erweiterter Text.
    """
    text = raw_text.strip()
    text = apply_corrections(text)
    text = apply_keyword_expansions(text)
    return text


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def build_initial_prompt(vocabulary: list) -> str:
    """
    Erstellt den initial_prompt fuer Whisper aus der Custom-Vocabulary-Liste
    und dem zusaetzlichen Prompt aus der Konfiguration.
    """
    base_prompt = INITIAL_PROMPT_EXTRA or ""
    if vocabulary:
        terms = ", ".join(vocabulary)
        vocab_part = f" In diesem Gespräch nutzen wir Fachbegriffe wie {terms}."
        return base_prompt + vocab_part
    return base_prompt or "Dies ist eine saubere Transkription auf Deutsch, die korrekte Satzzeichen und Grammatik verwendet."


def get_timestamp() -> str:
    """Gibt den aktuellen Zeitstempel im konfigurierten Format zurueck."""
    return datetime.datetime.now().strftime(TIMESTAMP_FORMAT)


def print_transcription(text: str) -> None:
    """
    Gibt eine abgeschlossene Transkription mit Zeitstempel auf der Konsole aus.

    Args:
        text: Verarbeiteter Transkriptionstext.
    """
    if not text:
        return

    timestamp = get_timestamp()
    print(OUTPUT_SEPARATOR)
    print(f"[{timestamp}] {text}")


def print_partial(text: str) -> None:
    """
    Gibt eine partielle (noch nicht abgeschlossene) Transkription aus.
    Ueberschreibt die aktuelle Zeile ohne Newline.

    Args:
        text: Partieller Transkriptionstext.
    """
    if text:
        # \r springt an den Zeilenanfang und ueberschreibt vorherigen Text
        print(f"\r  ... {text:<80}", end="", flush=True)


# =============================================================================
# TRANSKRIPTIONS-CALLBACKS
# =============================================================================

def on_transcription_complete(text: str) -> None:
    """
    Callback: wird aufgerufen wenn eine Aussage vollstaendig transkribiert wurde.
    - Post-Processing (Korrekturen + Expansionen)
    - Ausgabe in Konsole
    - Text an Cursor-Position tippen (wenn TYPE_INTO_CURSOR aktiv)
    - In Datei schreiben (wenn OUTPUT_FILE gesetzt)
    """
    # Partielle Anzeige loeschen
    print("\r" + " " * 90 + "\r", end="", flush=True)

    # Post-Processing anwenden
    processed = process_text(text)

    if not processed:
        return

    # Konsole
    print_transcription(processed)

    # In Datei schreiben
    if _output_file:
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        _output_file.write(f"[{timestamp}] {processed}\n")
        _output_file.flush()

    # An Cursor-Position tippen (via Zwischenablage fuer Umlaut-Unterstuetzung)
    if TYPE_INTO_CURSOR:
        pyperclip.copy(processed + " ")
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")


def on_transcription_partial(text: str) -> None:
    """
    Callback: wird aufgerufen waehrend der Transkription (Echtzeit-Vorschau).

    Args:
        text: Partieller transkribierter Text (noch nicht abgeschlossen).
    """
    print_partial(text)


def on_recording_start() -> None:
    """Callback: Mikrofon-Aufnahme hat begonnen."""
    # Optional: akustisches oder visuelles Signal
    pass


def on_recording_stop() -> None:
    """Callback: Mikrofon-Aufnahme wurde beendet, Transkription beginnt."""
    pass


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def main() -> None:
    """
    Initialisiert das Live-Transkriptionssystem und startet die Hauptschleife.

    Ablauf:
        1. Konfiguration ausgeben
        2. initial_prompt und hotwords aus Vocabulary erstellen
        3. AudioToTextRecorder mit faster-whisper (GPU, float16) initialisieren
        4. Dauerhaft auf Spracheingabe warten und transkribieren
        5. Bei Strg+C sauber beenden
    """

    # --- Startup-Informationen ausgeben ---
    print("=" * 60)
    print("  Live-Transkriptionssystem")
    print("  faster-whisper + RTX 5080 (Blackwell float16)")
    print("=" * 60)
    print(f"  Modell:       {MODEL_SIZE}")
    print(f"  Compute-Typ:  {COMPUTE_TYPE}")
    print(f"  Geraet:       {DEVICE}:{GPU_DEVICE_INDEX}")
    print(f"  Sprache:      {LANGUAGE}")
    print(f"  Vokabular:    {len(CUSTOM_VOCABULARY)} Begriffe")
    print(f"  Expansionen:  {len(KEYWORD_EXPANSIONS)} Eintraege")
    print(f"  Korrekturen:  {len(CORRECTIONS)} Regeln")
    print("=" * 60)
    print()

    # --- Prompt aufbauen ---
    initial_prompt = build_initial_prompt(CUSTOM_VOCABULARY)

    if initial_prompt:
        print(f"  Initial Prompt: {initial_prompt[:80]}...")

    # --- RealtimeSTT importieren ---
    try:
        from RealtimeSTT import AudioToTextRecorder
    except ImportError:
        print("[FEHLER] RealtimeSTT nicht installiert.")
        print("  Installieren: pip install RealtimeSTT")
        sys.exit(1)

    # --- Recorder konfigurieren und starten ---
    print("\n  Initialisiere AudioToTextRecorder...")
    print("  (Beim ersten Start wird das Whisper-Modell heruntergeladen.)")
    print("  Bitte warten...\n")

    recorder_config = {
        # Modell-Konfiguration
        "model": MODEL_SIZE,
        "language": LANGUAGE,
        "compute_type": COMPUTE_TYPE,
        "device": DEVICE,
        "gpu_device_index": GPU_DEVICE_INDEX,
        "beam_size": BEAM_SIZE,

        # Prompt für Kontext und Formatierung
        "initial_prompt": initial_prompt,

        # Einstellungen für Groß-/Kleinschreibung und Satzenden
        "ensure_sentence_starting_uppercase": True,
        "ensure_sentence_ends_with_period": True,

        # Audio / VAD (Voice Activity Detection)
        "silero_sensitivity": SILERO_SENSITIVITY,
        "min_length_of_recording": MIN_LENGTH_OF_RECORDING,
        "pre_recording_buffer_duration": PRE_RECORDING_BUFFER_DURATION,

        # Echtzeit-Transkriptions-Vorschau: Hauptmodell wiederverwenden
        "enable_realtime_transcription": True,
        "use_main_model_for_realtime": True,
        "realtime_processing_pause": 0.2,

        # Callbacks
        "on_realtime_transcription_update": on_transcription_partial,
        "on_recording_start": on_recording_start,
        "on_recording_stop": on_recording_stop,
    }

    try:
        recorder = AudioToTextRecorder(**recorder_config)
    except Exception as e:
        print(f"[FEHLER] Recorder konnte nicht initialisiert werden: {e}")
        print()
        print("  Haeufige Ursachen:")
        print("  - Kein Mikrofon gefunden oder Zugriffsrechte fehlen")
        print("  - CUDA/VRAM-Problem: compute_type in config.py pruefen")
        print("  - Abhaengigkeit fehlt: pip install RealtimeSTT faster-whisper")
        sys.exit(1)

    # --- Signal-Handler fuer sauberes Beenden ---
    def handle_shutdown(sig, frame):
        print("\n\n  [System] Beende Transkription... (Strg+C erkannt)")
        try:
            recorder.stop()
        except Exception:
            pass
        print("  [System] Auf Wiedersehen.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)

    # --- Bereit ---
    print(OUTPUT_SEPARATOR)
    print("  System bereit. Sprechen Sie ins Mikrofon.")
    print("  Beenden mit Strg+C")
    print(OUTPUT_SEPARATOR)
    print()

    # --- Haupt-Transkriptionsschleife ---
    # recorder.text() blockiert bis eine vollstaendige Aussage erkannt wurde
    # und ruft dann den Callback auf. Die Schleife sorgt fuer kontinuierliche
    # Verarbeitung weiterer Aussagen.
    while True:
        try:
            recorder.text(on_transcription_complete)
        except KeyboardInterrupt:
            handle_shutdown(None, None)
        except Exception as e:
            print(f"\n  [WARNUNG] Transkriptionsfehler: {e}")
            print("  Weiter mit naechster Aufnahme...")
            continue


if __name__ == "__main__":
    main()