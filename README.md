# WHISPER — Live-Transkriptionssystem

Echtzeit-Spracherkennung auf Deutsch mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper) und GPU-Beschleunigung. Transkribierter Text wird direkt an die aktuelle Cursor-Position getippt und parallel in eine Datei geschrieben. Alle Einstellungen sind über eine lokale Web-UI konfigurierbar.

## Hardware

Entwickelt und getestet auf:
- **GPU:** NVIDIA RTX 5080 (Blackwell, sm_120)
- **CPU:** AMD Ryzen 7 7800X3D
- **OS:** Windows 11 Pro

## Features

- Echtzeit-Transkription via Mikrofon (niedrige Latenz)
- GPU-Beschleunigung (CUDA, float16 — Pflicht für Blackwell)
- Text direkt an Cursor-Position tippen (via Zwischenablage)
- Transkription in Datei speichern
- **Batch-Verarbeitung** von Audio-/Videodateien über die Web-UI
- Custom Vocabulary (Fachbegriffe via `initial_prompt`)
- Keyword-Expansion: Abkürzungen werden automatisch ausgeschrieben
- Regelbasierte Korrekturen (Regex)
- **Web-UI** zur Konfiguration ohne Code-Änderungen (NiceGUI, Port 8080)

## Voraussetzungen

- Python 3.11 (nicht 3.12/3.13 — PyTorch cu128 Kompatibilität)
- CUDA 12.8
- NVIDIA-Treiber ≥ 570

## Installation

```powershell
# 1. Repository klonen
git clone https://github.com/Techmods/WHISPER.git
cd WHISPER

# 2. Virtuelle Umgebung mit Python 3.11
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# 3. PyTorch mit CUDA 12.8 ZUERST installieren
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Abhängigkeiten
pip install ctranslate2>=4.6.3 faster-whisper RealtimeSTT sounddevice numpy pyautogui pyperclip nicegui
```

## Starten

### Option A — Web-UI (empfohlen)

```powershell
.\venv\Scripts\Activate.ps1
python whisper_ui.py
```

Der Browser öffnet automatisch auf `http://localhost:8080`. Über die UI können alle Einstellungen angepasst, die Transkription gestartet/gestoppt und der Live-Output direkt im Browser verfolgt werden.

Alternativ: **`startui.bat` doppelklicken** — startet Python unsichtbar im Hintergrund, beendet automatisch laufende Instanzen und öffnet den Browser nach 3 Sekunden.

### Option B — direkt im Terminal

```powershell
.\venv\Scripts\Activate.ps1
python transcription_system.py
```

Fokus auf das gewünschte Textfeld setzen — transkribierter Text wird dort automatisch eingefügt. **Beenden:** `Strg+C`

Alternativ: `startwhisper.bat` doppelklicken.

Beim ersten Start wird das Whisper-Modell heruntergeladen (je nach Modell 75 MB – 3 GB).

## Web-UI

Die Web-UI läuft lokal auf Port 8080 und ermöglicht die vollständige Konfiguration ohne Code-Änderungen.

| Bereich | Funktion |
|---|---|
| Modell | Dropdown mit VRAM-Info für alle unterstützten Modelle |
| Compute Type | float16 / bfloat16 / int8_float16 — mit Blackwell-Warnung |
| Sprache | Deutsch, Englisch, weitere oder automatische Erkennung |
| Beam Size | Slider 1–10 (Geschwindigkeit vs. Genauigkeit) |
| Initial Prompt | Freitext für Kontext, Stil und Formatierungsvorgaben |
| Audio-Eingabe | Dropdown mit allen verfügbaren Mikrofonen |
| VAD | Voice Activity Detection toggle + Silero-Empfindlichkeit |
| Custom Vocabulary | Fachbegriffe als Chips (hinzufügen/entfernen) |
| Keyword-Expansionen | Tabelle: Abkürzung → Volltext |
| Korrekturen | Tabelle: Von (Regex) → Nach |
| Ausgabe | Cursor-Tippen toggle + Ausgabedatei mit Datei-Explorer-Dialog |
| Steuerung | Start/Stop-Button, Live-Transkriptionslog |
| **Dateiverarbeitung** | Audio-/Videodateien in Warteschlange legen und asynchron transkribieren |

Konfigurationsänderungen werden per **Speichern**-Button in `config.py` geschrieben (alle Kommentare bleiben erhalten). Ein Neustart der Transkription ist danach erforderlich.

## Batch-Verarbeitung

Über den Tab **Dateiverarbeitung** in der Web-UI können Audio- und Videodateien asynchron transkribiert werden. Die Verarbeitung läuft über `batch_transcriber.py` als separaten Prozess — das Live-Mikrofon wird dabei nicht beeinflusst.

Unterstützte Formate: alle von faster-whisper unterstützten Audio/Video-Container (mp3, wav, m4a, mp4, mkv, …).

## Modelle

### Offizielle OpenAI-Modelle (mehrsprachig)

| Modell | VRAM | Sprachen | Geschwindigkeit | Qualität |
|---|---|---|---|---|
| `tiny` | 75 MB | 99 Sprachen | extrem schnell | niedrig |
| `base` | 150 MB | 99 Sprachen | sehr schnell | gering |
| `small` | 500 MB | 99 Sprachen | schnell | mittel |
| `medium` | 1.5 GB | 99 Sprachen | mittel | gut |
| `large-v2` | 3 GB | 99 Sprachen | langsam | sehr gut |
| `large-v3` | 3 GB | 99 Sprachen | langsam | beste Qualität |

### Turbo-Modelle (schnell + mehrsprachig)

| Modell | VRAM | Sprachen | Geschwindigkeit | Qualität |
|---|---|---|---|---|
| `deepdml/faster-whisper-large-v3-turbo-ct2` | 1.6 GB | 99 Sprachen | ~8x schneller als large-v3 | sehr gut |

### Deutsch-finetuned

| Modell | VRAM | Sprachen | Geschwindigkeit | Qualität |
|---|---|---|---|---|
| `MR-Eder/faster-whisper-large-v3-turbo-german` | 1.6 GB | nur Deutsch | sehr schnell | sehr gut |
| `Primeline/whisper-large-v3-turbo-german` | 1.6 GB | nur Deutsch | sehr schnell | sehr gut |
| `primeline/whisper-large-v3-german` | 3 GB | nur Deutsch | langsam | exzellent |

> **Empfehlung:** `deepdml/faster-whisper-large-v3-turbo-ct2` für den besten Kompromiss aus Geschwindigkeit und Qualität. Deutsch-only-Modelle nur wenn ausschließlich Deutsch transkribiert wird.

> **distil-large-v3 ist englisch-only** — nicht für Deutsch geeignet.

Alle Modelle sind in `config.py` als Kommentare hinterlegt und in der Web-UI als Dropdown auswählbar.

## Konfiguration

Alle Parameter in `config.py` — entweder manuell oder über die Web-UI:

| Parameter | Beschreibung | Standard |
|---|---|---|
| `MODEL_SIZE` | Whisper-Modell (siehe Tabelle oben) | `deepdml/faster-whisper-large-v3-turbo-ct2` |
| `COMPUTE_TYPE` | **Für Blackwell zwingend** `float16` oder `bfloat16` | `float16` |
| `LANGUAGE` | Sprache (`de`, `en`, `auto`, …) | `de` |
| `BEAM_SIZE` | Dekodierungsqualität (1 = schnell, 10 = genau) | `5` |
| `INITIAL_PROMPT_EXTRA` | Zusätzlicher Kontext/Stil für Whisper | `""` |
| `INPUT_DEVICE_INDEX` | Mikrofon-Index (`None` = System-Standard) | `None` |
| `VAD_ENABLED` | Voice Activity Detection aktivieren | `True` |
| `SILERO_SENSITIVITY` | VAD-Empfindlichkeit (0.0–1.0) | `0.4` |
| `CUSTOM_VOCABULARY` | Fachbegriffe für bessere Erkennung | Liste |
| `KEYWORD_EXPANSIONS` | Abkürzung → Volltext (Post-Processing) | Dict |
| `CORRECTIONS` | Regex-Korrekturen für Erkennungsfehler | Dict |
| `TYPE_INTO_CURSOR` | Text an Cursor-Position tippen | `True` |
| `OUTPUT_FILE` | Pfad zur Ausgabedatei (`None` = deaktiviert) | `transkription.txt` |

## Blackwell-Hinweis (RTX 5080)

CTranslate2 ≥ 4.6.3 unterstützt Blackwell (sm_120), hat aber INT8 explizit deaktiviert. `compute_type` muss `float16` oder `bfloat16` sein — INT8 und int8_float16 werfen einen Fehler.

## GPU testen

```powershell
python test_gpu.py
```
