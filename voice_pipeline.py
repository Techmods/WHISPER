# =============================================================================
# voice_pipeline.py — Voice-to-Voice Orchestrator
#
# Verbindet STT (transcription_system.py / process_manager.py),
# LLM (OpenRouter API via llm_client.py) und TTS (tts_process.py).
#
# Architektur:
#   CPU: STT via faster-whisper (process_manager.py / transcription_system.py)
#   Netz: LLM via OpenRouter API (llm_client.py, asyncio Streaming)
#   GPU:  TTS via tts_process.py (eigener Subprocess)
#
# Barge-In-Logik:
#   Wenn STT während einer TTS-Ausgabe neue Sprache erkennt,
#   wird sofort ein "interrupt"-Kommando an den TTS-Subprocess gesendet
#   und der laufende LLM-Stream abgebrochen.
#
# Lifecycle-States:
#   "offline" | "starting" | "ready" | "listening" | "thinking" | "speaking" | "stopping"
#
# Verwendung:
#   from voice_pipeline import VoicePipeline
#   pipeline = VoicePipeline(config)
#   await pipeline.start()
#   ...
#   await pipeline.stop()
# =============================================================================

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable

PROJECT_DIR = Path(__file__).parent
VENV_PYTHON = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
TTS_SCRIPT = PROJECT_DIR / "tts_process.py"


# ---------------------------------------------------------------------------
# State-Management (gleiche Konvention wie process_manager.py)
# ---------------------------------------------------------------------------

_state: str = "offline"
_state_callbacks: list[Callable[[str], None]] = []
_log_callbacks: list[Callable[[str], None]] = []
_transcript_callbacks: list[Callable[[str, str], None]] = []  # (role, text)


def get_state() -> str:
    return _state


def on_state_change(cb: Callable[[str], None]) -> None:
    _state_callbacks.append(cb)


def on_log(cb: Callable[[str], None]) -> None:
    _log_callbacks.append(cb)


def on_transcript(cb: Callable[[str, str], None]) -> None:
    """Callback mit (role, text) — role ist 'user' oder 'assistant'."""
    _transcript_callbacks.append(cb)


def _set_state(new_state: str) -> None:
    global _state
    if new_state == _state:
        return
    _state = new_state
    for cb in _state_callbacks:
        try:
            cb(new_state)
        except Exception:
            pass


def _log(line: str) -> None:
    for cb in _log_callbacks:
        try:
            cb(line)
        except Exception:
            pass


def _emit_transcript(role: str, text: str) -> None:
    for cb in _transcript_callbacks:
        try:
            cb(role, text)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------

class VoicePipeline:
    """
    Orchestriert die Voice-to-Voice-Pipeline:
      STT-Transkript → LLM-Stream → TTS-Satz-Queue → Audio-Ausgabe
    """

    def __init__(
        self,
        openrouter_api_key: str,
        openrouter_model: str,
        llm_system_prompt: str,
        tts_enabled: bool = True,
        tts_model: str = "Qwen/Qwen2-Audio-7B-Instruct",
        tts_voice: str = "default",
        tts_output_device_index: int | None = None,
        tts_volume: float = 0.9,
        tts_barge_in_enabled: bool = True,
    ):
        self.api_key = openrouter_api_key
        self.model = openrouter_model
        self.system_prompt = llm_system_prompt
        self.tts_enabled = tts_enabled
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_output_device_index = tts_output_device_index
        self.tts_volume = tts_volume
        self.tts_barge_in_enabled = tts_barge_in_enabled

        # Konversations-History (OpenAI-Format)
        self._history: list[dict] = []

        # TTS-Subprocess-Handle
        self._tts_proc: asyncio.subprocess.Process | None = None
        self._tts_ready = asyncio.Event()

        # LLM-Interrupt: wird gefeuert wenn Barge-In erkannt wird
        self._llm_interrupt = asyncio.Event()

        # Laufende Pipeline-Task
        self._pipeline_task: asyncio.Task | None = None

        # Gibt an ob gerade TTS abspielt (für Barge-In-Entscheidung)
        self._tts_speaking = False

        # Gibt an ob Pipeline läuft
        self._running = False

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Startet die Pipeline (TTS-Subprocess + STT-Hook)."""
        if self._running:
            return

        _set_state("starting")
        self._running = True

        if self.tts_enabled:
            await self._start_tts_process()

        _set_state("ready")
        _log("[Pipeline] Voice-to-Voice Pipeline bereit.")

    async def stop(self) -> None:
        """Stoppt die Pipeline sauber."""
        if not self._running:
            return

        _set_state("stopping")
        self._running = False

        # Laufenden LLM-Stream abbrechen
        self._llm_interrupt.set()

        # TTS-Subprocess beenden
        await self._stop_tts_process()

        # Laufenden Pipeline-Task abbrechen
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass

        self._history.clear()
        _set_state("offline")
        _log("[Pipeline] Pipeline gestoppt.")

    # ------------------------------------------------------------------
    # Haupt-Einstiegspunkt: neue STT-Transkription eingekommen
    # ------------------------------------------------------------------

    async def handle_transcript(self, text: str) -> None:
        """
        Wird aufgerufen wenn STT eine vollständige Aussage erkannt hat.
        Löst Barge-In aus (falls TTS spricht) und startet LLM-Anfrage.
        """
        if not self._running:
            return

        text = text.strip()
        if not text:
            return

        # Barge-In: laufende Ausgabe unterbrechen
        if self._tts_speaking and self.tts_barge_in_enabled:
            _log("[Pipeline] Barge-In: unterbreche TTS-Ausgabe.")
            await self._interrupt_tts()

        # Laufenden LLM-Stream abbrechen
        self._llm_interrupt.set()
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except (asyncio.CancelledError, Exception):
                pass

        # User-Aussage loggen
        _emit_transcript("user", text)
        _log(f"[User] {text}")

        # Neuen Pipeline-Durchlauf starten
        self._llm_interrupt.clear()
        self._pipeline_task = asyncio.create_task(self._run_pipeline(text))

    # ------------------------------------------------------------------
    # Pipeline-Durchlauf
    # ------------------------------------------------------------------

    async def _run_pipeline(self, user_text: str) -> None:
        """Einzelner Pipeline-Durchlauf: LLM-Stream → TTS-Queue."""
        from llm_client import stream_llm_response, build_messages

        _set_state("thinking")

        messages = await build_messages(self.system_prompt, self._history, user_text)

        assistant_text_parts: list[str] = []

        async def on_sentence(sentence: str) -> None:
            """Wird für jeden fertigen LLM-Satz aufgerufen."""
            if self._llm_interrupt.is_set():
                return
            _log(f"[TTS←LLM] {sentence}")
            assistant_text_parts.append(sentence)

            if self.tts_enabled:
                _set_state("speaking")
                self._tts_speaking = True
                await self._send_to_tts(sentence)
            else:
                # Kein TTS: Antwort direkt als Transcript ausgeben
                _emit_transcript("assistant", sentence)

        try:
            full_response = await stream_llm_response(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                on_sentence=on_sentence,
                interrupt_event=self._llm_interrupt,
            )
        except asyncio.CancelledError:
            _log("[Pipeline] LLM-Stream abgebrochen (Barge-In).")
            return
        except Exception as e:
            _log(f"[Pipeline] LLM-Fehler: {e}")
            _set_state("ready")
            return

        # Vollständige Antwort zur History hinzufügen
        complete_response = full_response.strip()
        if complete_response:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": complete_response})
            _emit_transcript("assistant", complete_response)
            # History auf max. 20 Turns begrenzen (10 User + 10 Assistant)
            if len(self._history) > 20:
                self._history = self._history[-20:]

        if not self._tts_speaking:
            _set_state("ready")

    # ------------------------------------------------------------------
    # TTS-Subprocess-Verwaltung
    # ------------------------------------------------------------------

    async def _start_tts_process(self) -> None:
        """Startet tts_process.py als Subprocess."""
        _log(f"[Pipeline] Starte TTS-Subprocess: {self.tts_model}")

        python_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

        cmd = [
            python_exe, "-u", str(TTS_SCRIPT),
            "--model", self.tts_model,
            "--voice", self.tts_voice,
            "--volume", str(self.tts_volume),
        ]
        if self.tts_output_device_index is not None:
            cmd += ["--device-index", str(self.tts_output_device_index)]

        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

        try:
            self._tts_proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_DIR),
                env=env,
            )
        except Exception as e:
            _log(f"[Pipeline] TTS-Subprocess konnte nicht gestartet werden: {e}")
            self._tts_proc = None
            return

        # Stdout-Reader-Task starten
        asyncio.create_task(self._read_tts_stdout())

        # Warten bis TTS-Modell geladen ist (max. 120 s)
        try:
            await asyncio.wait_for(self._tts_ready.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            _log("[Pipeline] TTS-Modell konnte nicht rechtzeitig geladen werden.")

    async def _stop_tts_process(self) -> None:
        """Stoppt den TTS-Subprocess sauber."""
        if self._tts_proc is None:
            return
        try:
            await self._send_tts_cmd({"cmd": "quit"})
            try:
                await asyncio.wait_for(self._tts_proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._tts_proc.kill()
        except Exception:
            pass
        self._tts_proc = None

    async def _interrupt_tts(self) -> None:
        """Sendet Interrupt-Kommando an den TTS-Subprocess."""
        await self._send_tts_cmd({"cmd": "interrupt"})
        self._tts_speaking = False

    async def _send_to_tts(self, text: str) -> None:
        """Sendet einen Satz an den TTS-Subprocess."""
        await self._send_tts_cmd({"cmd": "speak", "text": text})

    async def _send_tts_cmd(self, cmd: dict) -> None:
        """Schreibt ein JSON-Kommando in stdin des TTS-Subprozesses."""
        if self._tts_proc is None or self._tts_proc.stdin is None:
            return
        try:
            line = json.dumps(cmd, ensure_ascii=False) + "\n"
            self._tts_proc.stdin.write(line.encode("utf-8"))
            await self._tts_proc.stdin.drain()
        except Exception as e:
            _log(f"[Pipeline] TTS-Schreibfehler: {e}")

    async def _read_tts_stdout(self) -> None:
        """Liest TTS-Subprocess-Stdout und reagiert auf Events."""
        if self._tts_proc is None or self._tts_proc.stdout is None:
            return

        try:
            async for raw in self._tts_proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    _log(f"[TTS] {line}")
                    continue

                evt = event.get("event", "")
                text = event.get("text", "")

                if evt == "ready":
                    _log("[TTS] Modell geladen, TTS bereit.")
                    self._tts_ready.set()
                elif evt == "started":
                    _log(f"[TTS] Spiele: {text[:60]}...")
                    self._tts_speaking = True
                elif evt == "done":
                    self._tts_speaking = False
                    if get_state() == "speaking":
                        _set_state("ready")
                elif evt == "interrupted":
                    self._tts_speaking = False
                    if get_state() == "speaking":
                        _set_state("ready")
                elif evt == "error":
                    detail = event.get("detail", "")
                    _log(f"[TTS] Fehler: {detail}")
                    self._tts_speaking = False
                elif evt in ("loading_model", "model_ready", "model_fallback"):
                    backend = event.get("backend", "")
                    _log(f"[TTS] {evt}: {backend or event.get('model', '')}")

        except Exception:
            pass

        # Subprocess beendet
        self._tts_proc = None
        if self._running:
            _log("[TTS] Subprocess unerwartet beendet.")


# ---------------------------------------------------------------------------
# Modul-Level Singleton (für process_manager.py / whisper_ui.py)
# ---------------------------------------------------------------------------

_pipeline: VoicePipeline | None = None


def get_pipeline() -> VoicePipeline | None:
    return _pipeline


def is_pipeline_running() -> bool:
    return _pipeline is not None and _pipeline.is_running()


async def start_pipeline(
    openrouter_api_key: str,
    openrouter_model: str,
    llm_system_prompt: str,
    tts_enabled: bool = True,
    tts_model: str = "Qwen/Qwen2-Audio-7B-Instruct",
    tts_voice: str = "default",
    tts_output_device_index: int | None = None,
    tts_volume: float = 0.9,
    tts_barge_in_enabled: bool = True,
) -> bool:
    """Startet die globale Pipeline-Instanz. Gibt False zurück wenn bereits läuft."""
    global _pipeline

    if is_pipeline_running():
        return False

    _pipeline = VoicePipeline(
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        llm_system_prompt=llm_system_prompt,
        tts_enabled=tts_enabled,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_output_device_index=tts_output_device_index,
        tts_volume=tts_volume,
        tts_barge_in_enabled=tts_barge_in_enabled,
    )
    await _pipeline.start()
    return True


async def stop_pipeline() -> bool:
    """Stoppt die globale Pipeline-Instanz."""
    global _pipeline

    if not is_pipeline_running():
        return False

    await _pipeline.stop()
    _pipeline = None
    return True


async def handle_stt_transcript(text: str) -> None:
    """
    Wird von process_manager.py aufgerufen wenn STT eine vollständige
    Transkription produziert hat.
    Leitet den Text an die aktive Pipeline weiter.
    """
    if _pipeline is not None and _pipeline.is_running():
        await _pipeline.handle_transcript(text)
