---
name: WHISPER-Agent
description: Instruktionen für den KI-Assistenten (Gemini/Antigravity) für das WHISPER Live-Transkriptionssystem.
---

# WHISPER KI-Assistent

Diese Datei ersetzt die alten Claude-Instruktionen und dient als primäre Richtlinie für mich (und zukünftige KI-Assistenten) bei der Arbeit an diesem Projekt.

## Projektkontext (`WHISPER`)
Das Projekt ist ein **Live-Transkriptionssystem** auf Basis von `faster-whisper`.
Es zeichnet Audiodaten auf, transkribiert diese lokal (Echtzeit) und tippt sie direkt an die aktuelle Cursor-Position oder speichert sie in einer Textdatei.

### Technische Umgebung
- **OS:** Windows 11
- **Python:** 3.11 (zwingend erforderlich wegen PyTorch cu128 Kompatibilität)
- **CUDA:** 12.8
- **GPU Limitierung (Blackwell / RTX 5080):** `compute_type` muss zwingend `float16` oder `bfloat16` sein, INT8 wirft bei CTranslate2 ≥ 4.6.3 auf der RTX 5080 Fehler.
- **UI:** NiceGUI (Lokaler Webserver auf Port 8080 zur Konfiguration der `config.py`)

## Instruktionen für mich

1. **Sprache:** Ich antworte standardmäßig auf Deutsch (wie vom Nutzer gewünscht).
2. **Technologien:** Bei Installationen von Python-Paketen achte ich darauf, dass sie mit Python 3.11 und CUDA 12.8 / RTX 5080 kompatibel sind (z.B. der oben genannte `compute_type` Bug).
3. **Konfiguration:** Alle Anpassungen der Nutzer-Einstellungen werden in der `config.py` vorgenommen, welche auch von der NiceGUI-Web-App (in `whisper_ui.py`) gelesen wird. Wir müssen darauf achten, die Kommentare dort intakt zu lassen.
4. **Kein Claude-Restbestand:** Wir nutzen nicht mehr `CLAUDE.md` oder `.claude/`, stattdessen bauen wir auf dieses Projekt-Setup (und Skills).

Diesen Kontext werde ich ab jetzt für alle künftigen Erweiterungen und Fehlerbehebungen im WHISPER-Projekt verwenden.
