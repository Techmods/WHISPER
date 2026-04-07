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
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

# Rechentyp: ZWINGEND "float16" für RTX 5080 (Blackwell sm_120)
# INT8 ist für Blackwell (sm_120) in CTranslate2 >= 4.6.3 deaktiviert.
COMPUTE_TYPE = "float16"

# Inferenzgerät: "cuda" für GPU, "cpu" als Fallback
DEVICE = "cuda"

# GPU-Index (bei mehreren GPUs ggf. anpassen)
GPU_DEVICE_INDEX = 0

# Transkriptionssprache (ISO 639-1): "de" für Deutsch, "en" für Englisch
# None = automatische Erkennung (langsamer)
LANGUAGE = "de"

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
    # Beispiel-Einträge — an eigene Fachdomäne anpassen:
    "Transkription",
    "Echtzeit",
    "Spracherkennung",
    "Whisper",
    "CTranslate2",
    "CUDA",
    "Blackwell",
    "RTX 5080",
    # Eigene Fachbegriffe hier eintragen:
    # "MeinFachbegriff",
    # "Firmenname GmbH",
    # "Produktname XL3000",
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
    # Beispiel-Expansionen — an eigene Fachdomäne anpassen:
    "ki": "Künstliche Intelligenz (KI)",
    "ml": "Machine Learning (ML)",
    "nlp": "Natural Language Processing (NLP)",
    "api": "API (Application Programming Interface)",
    # Eigene Expansionen hier eintragen:
    # "vp": "Vizepräsident",
    # "gf": "Geschäftsführer",
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
    # Beispiel-Korrekturen — an eigene Fehler anpassen:
    r"\bWhisba\b": "Whisper",
    r"\bBlackval\b": "Blackwell",
    r"\bSi Transleyt\b": "CTranslate2",
    # Eigene Korrekturen hier eintragen:
    # r"\bFalschSchreibung\b": "KorrekteSchreibung",
}

# ---------------------------------------------------------------------------
# AUDIO-PARAMETER (RealtimeSTT)
# ---------------------------------------------------------------------------

# Stille-Schwelle nach der eine Aufnahme als abgeschlossen gilt (Sekunden)
SILERO_SENSITIVITY = 0.4

# Minimale Aufnahmedauer damit eine Transkription ausgelöst wird (Sekunden)
MIN_LENGTH_OF_RECORDING = 0.5

# Pre-Recording-Puffer: wie viel Audio vor Sprachbeginn erhalten bleibt (Sekunden)
PRE_RECORDING_BUFFER_DURATION = 0.2

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
OUTPUT_FILE = r"C:\DEV\WHISPER\transkription.txt"
