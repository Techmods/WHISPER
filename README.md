# WHISPER — Live-Transkriptionssystem

Echtzeit-Spracherkennung auf Deutsch mit [faster-whisper](https://github.com/SYSTRAN/faster-whisper) und GPU-Beschleunigung. Transkribierter Text wird direkt an die aktuelle Cursor-Position getippt und parallel in eine Datei geschrieben.

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
- Custom Vocabulary (Fachbegriffe via `initial_prompt`)
- Keyword-Expansion: Abkürzungen werden automatisch ausgeschrieben
- Regelbasierte Korrekturen (Regex)

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
pip install ctranslate2>=4.6.3 faster-whisper RealtimeSTT sounddevice numpy pyautogui pyperclip
```

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
| `Primeline/whisper-large-v3-turbo-german` | 1.6 GB | nur Deutsch | sehr schnell | sehr gut |
| `primeline/whisper-large-v3-german` | 3 GB | nur Deutsch | langsam | exzellent |

> **Empfehlung:** `deepdml/faster-whisper-large-v3-turbo-ct2` für den besten Kompromiss aus Geschwindigkeit und Qualität bei Deutsch. Deutsch-only-Modelle nur wenn ausschließlich Deutsch transkribiert wird.

> **distil-large-v3 ist englisch-only** — nicht für Deutsch geeignet.

Modell in `config.py` ändern — alle Optionen sind dort als Kommentar hinterlegt.

## Konfiguration

Alle Parameter in `config.py` anpassen:

| Parameter | Beschreibung | Standard |
|---|---|---|
| `MODEL_SIZE` | Whisper-Modell (siehe Tabelle oben) | `large-v3` |
| `COMPUTE_TYPE` | **Für Blackwell zwingend** `float16` | `float16` |
| `LANGUAGE` | Sprache (`de`, `en`, …) | `de` |
| `BEAM_SIZE` | Dekodierungsqualität (1 = schnell, 5 = genau) | `2` |
| `CUSTOM_VOCABULARY` | Fachbegriffe für bessere Erkennung | Liste |
| `KEYWORD_EXPANSIONS` | Abkürzung → Volltext (Post-Processing) | Dict |
| `CORRECTIONS` | Regex-Korrekturen für Erkennungsfehler | Dict |
| `TYPE_INTO_CURSOR` | Text an Cursor-Position tippen | `True` |
| `OUTPUT_FILE` | Pfad zur Ausgabedatei (`None` = deaktiviert) | `transkription.txt` |

## Starten

```powershell
.\venv\Scripts\Activate.ps1
python transcription_system.py
```

Beim ersten Start wird das Whisper-Modell heruntergeladen (~3 GB für `large-v3`). Danach startet das System sofort.

Fokus auf das gewünschte Textfeld setzen — transkribierter Text wird dort automatisch eingefügt.

**Beenden:** `Strg+C`

## Blackwell-Hinweis (RTX 5080)

CTranslate2 ≥ 4.6.3 unterstützt Blackwell (sm_120), hat aber INT8 explizit deaktiviert. `compute_type="float16"` ist Pflicht — INT8 wirft einen Fehler.

## GPU testen

```powershell
python test_gpu.py
```
