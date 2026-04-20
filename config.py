# =============================================================================
# config.py — Konfigurationsdatei für das Live-Transkriptionssystem
#
# Alle benutzerdefinierten Parameter hier anpassen.
# Diese Datei wird von transcription_system.py importiert.
# =============================================================================

# ---------------------------------------------------------------------------
# MODELL-PARAMETER
# ---------------------------------------------------------------------------

# Whisper-Modell — Deutsch-kompatible Optionen (eine Zeile einkommentieren):
#
# --- Offizielle OpenAI-Modelle (mehrsprachig, inkl. Deutsch) ---
# MODEL_SIZE = "tiny"                                          # 75 MB   | alle Sprachen | sehr schnell, niedrige Qualität
# MODEL_SIZE = "base"                                          # 150 MB  | alle Sprachen | schnell
# MODEL_SIZE = "small"                                         # 500 MB  | alle Sprachen | gute Balance
# MODEL_SIZE = "medium"                                        # 1.5 GB  | alle Sprachen | gut
# MODEL_SIZE = "large-v2"                                      # 3 GB    | alle Sprachen | sehr gut
# MODEL_SIZE = "large-v3"                                      # 3 GB    | alle Sprachen | beste Qualität (Standard)
#
# --- Turbo (schnell + mehrsprachig) ---
# MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"    # 1.6 GB  | alle Sprachen | ~8x schneller als large-v3, empfohlen
#
# --- Deutsch-finetuned ---
# MODEL_SIZE = "Primeline/whisper-large-v3-turbo-german"      # 1.6 GB  | nur Deutsch   | optimiert für DE, sehr schnell
# MODEL_SIZE = "primeline/whisper-large-v3-german"            # 3 GB    | nur Deutsch   | beste DE-Qualität
#
MODEL_SIZE = 'deepdml/faster-whisper-large-v3-turbo-ct2'

# Rechentyp: ZWINGEND "float16" für RTX 5080 (Blackwell sm_120)
# INT8 ist für Blackwell (sm_120) in CTranslate2 >= 4.6.3 deaktiviert.
# Rechentyp-Optionen:
#   "float16"       — Pflicht für Blackwell (RTX 5080, sm_120)
#   "bfloat16"      — Alternative zu float16, ähnliche Geschwindigkeit
#   "int8_float16"  — Weniger VRAM, fast identische Genauigkeit (nicht für Blackwell)
#   "int8"          — NICHT für Blackwell (sm_120) — CTranslate2 >= 4.6.3 deaktiviert
COMPUTE_TYPE = 'float16'

# Inferenzgerät: "cuda" für GPU, "cpu" als Fallback
DEVICE = 'cuda'

# GPU-Index (bei mehreren GPUs ggf. anpassen)
GPU_DEVICE_INDEX = 0

# Transkriptionssprache (ISO 639-1): "de" für Deutsch, "en" für Englisch
# None = automatische Erkennung (langsamer)
LANGUAGE = 'de'

# Beam-Größe für die Dekodierung (höher = genauer, aber langsamer)
BEAM_SIZE = 5

# ---------------------------------------------------------------------------
# CUSTOM VOCABULARY
# ---------------------------------------------------------------------------
# Liste von Fachbegriffen, Eigennamen oder ungewöhnlichen Wörtern.
# Diese werden als initial_prompt und hotwords an Whisper übergeben,
# um die Erkennungsgenauigkeit für diese Begriffe zu verbessern.
#
# Hinweise:
# - Groß-/Kleinschreibung wie im Zieltext angeben
# - Abkürzungen und Aussprache-Varianten separat eintragen
# - Zu lange Listen (> 50 Begriffe) können die Genauigkeit verschlechtern
# ---------------------------------------------------------------------------
CUSTOM_VOCABULARY = [
    'Transkription',
    'Echtzeit',
    'Spracherkennung',
    'Whisper',
    'CTranslate2',
    'CUDA',
    'Blackwell',
    'RTX 5080',
    'Obsidian',
    'faster whisper',
    'claude',
    'claude code',
]

# ---------------------------------------------------------------------------
# KEYWORD-EXPANSIONEN
# ---------------------------------------------------------------------------
# Schlüssel → erweiterter Ersetzungstext (Post-Processing nach Transkription).
# Wenn ein Schlüsselwort (Groß-/Kleinschreibung wird ignoriert) im transkribierten
# Text vorkommt, wird es durch den Ersetzungstext ersetzt.
#
# Anwendungsfälle:
# - Abkürzungen ausschreiben: "KI" → "Künstliche Intelligenz"
# - Schreibweisen vereinheitlichen
# - Markennamen korrekt formatieren
#
# Hinweis: Die Reihenfolge der Anwendung ist von der dict-Reihenfolge abhängig.
# Längere/spezifischere Schlüssel vor kürzeren eintragen, um ungewollte
# Teilersetzungen zu vermeiden.
# ---------------------------------------------------------------------------
KEYWORD_EXPANSIONS = {
    'ki': 'Künstliche Intelligenz (KI)',
    'ml': 'Machine Learning (ML)',
    'nlp': 'Natural Language Processing (NLP)',
    'api': 'API (Application Programming Interface)',
}

# ---------------------------------------------------------------------------
# KORREKTUREN
# ---------------------------------------------------------------------------
# Falsche Schreibung → korrekte Schreibung (Post-Processing, regex-basiert).
# Whisper kann manche Wörter konsistent falsch transkribieren — hier beheben.
#
# Beispiele:
# - Homophone: "das" vs "dass" in bestimmten Kontexten
# - Eigennamen die Whisper falsch schreibt
# - Fachbegriffe mit ungewöhnlicher Schreibweise
#
# Schlüssel sind reguläre Ausdrücke (re.sub-kompatibel).
# Wert ist der Ersetzungstext (kann Rückreferenzgruppen enthalten).
# Matching ist standardmäßig case-insensitive (flags=re.IGNORECASE).
# ---------------------------------------------------------------------------
CORRECTIONS = {
    '\\bWhisba\\b': 'Whisper',
    '\\bBlackval\\b': 'Blackwell',
    '\\bSi Transleyt\\b': 'CTranslate2',
}

# ---------------------------------------------------------------------------
# INITIAL PROMPT & STILE
# ---------------------------------------------------------------------------
# Stil-Vorgabe (z.B. "standard", "code", "raw")
TRANSCRIPTION_STYLE_PRESET = 'standard'

# Gibt Whisper Kontext zur Formatierung, Interpunktion und Stil.
# Fachbegriffe aus CUSTOM_VOCABULARY werden automatisch angehängt.
# Leer lassen für automatischen Prompt aus CUSTOM_VOCABULARY.
INITIAL_PROMPT_EXTRA = 'Dies ist eine technische Diskussion in deutscher Sprache. Bitte achte auf korrekte Groß- und Kleinschreibung, Satzzeichen (Punkte, Kommas, Fragezeichen) und setze nach jedem vollständigen Gedanken einen Absatz.'

# ---------------------------------------------------------------------------
# AUDIO-PARAMETER (RealtimeSTT)
# ---------------------------------------------------------------------------

# Mikrofon-Eingabegerät (None = System-Standard)
# Index ermitteln: python -c "import sounddevice as sd; print(sd.query_devices())"
INPUT_DEVICE_INDEX = None

# VAD (Voice Activity Detection) via Silero aktivieren
VAD_ENABLED = True

# Zeit nach Sprach-Ende, bevor der Satz abgeschnitten wird ("Denkpausen-Dauer" in Sek)
POST_SPEECH_SILENCE_DURATION = 4

# Stille-Schwelle nach der eine Aufnahme als abgeschlossen gilt (Sekunden)
SILERO_SENSITIVITY = 0.4

# Minimale Aufnahmedauer damit eine Transkription ausgelöst wird (Sekunden)
MIN_LENGTH_OF_RECORDING = 0.5

# Pre-Recording-Puffer: wie viel Audio vor Sprachbeginn erhalten bleibt (Sekunden)
PRE_RECORDING_BUFFER_DURATION = 0.5

# Wake-Word-Erkennung deaktivieren (keine Aktivierungsvokabel benötigt)
WAKE_WORDS_SENSITIVITY = 0.0

# ---------------------------------------------------------------------------
# AUSGABE-PARAMETER
# ---------------------------------------------------------------------------

# Zeitstempelformat für die Konsolenausgabe
TIMESTAMP_FORMAT = "%H:%M:%S"

# Trennlinie für bessere Lesbarkeit der Ausgabe
OUTPUT_SEPARATOR = "-" * 60

# ---------------------------------------------------------------------------
# AUSGABE-AKTIONEN
# ---------------------------------------------------------------------------

# Text direkt an die aktuelle Cursor-Position tippen (via Zwischenablage + Strg+V)
# True = aktiv, False = deaktiviert
TYPE_INTO_CURSOR = True

# Transkriptionen in Datei schreiben (append). None = deaktiviert.
# Beispiel: OUTPUT_FILE = r"C:\DEV\WHISPER\transkription.txt"
OUTPUT_FILE = 'C:\\DEV\\WHISPER\\transkription.txt'