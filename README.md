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

## Konfiguration

Alle Parameter in `config.py` anpassen:

| Parameter | Beschreibung | Standard |
|---|---|---|
| `MODEL_SIZE` | Whisper-Modell (`large-v3`, `distil-large-v3`, `medium`) | `large-v3` |
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
