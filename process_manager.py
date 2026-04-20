# =============================================================================
# process_manager.py — asyncio Subprocess-Lifecycle für transcription_system.py
#
# start_transcription() startet transcription_system.py als Kindprozess.
# stop_transcription()  sendet CTRL_BREAK_EVENT, wartet 5s, dann kill().
# Stdout-Lines werden in einem Ringpuffer gespeichert (für ui.log()).
# =============================================================================

import asyncio
import collections
import os
import signal
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
VENV_PYTHON = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
SCRIPT = PROJECT_DIR / "transcription_system.py"

# Prozess-Handle (None = nicht gestartet)
_process: asyncio.subprocess.Process | None = None

# Ringpuffer für stdout-Zeilen (max. 500 Zeilen)
_log_buffer: collections.deque = collections.deque(maxlen=500)

# Callbacks die aufgerufen werden wenn eine neue Zeile ankommt
_log_callbacks: list = []


def is_running() -> bool:
    """Gibt True zurück wenn der Transkriptions-Prozess aktiv läuft."""
    return _process is not None and _process.returncode is None


def get_log_buffer() -> list:
    """Gibt den aktuellen Log-Puffer als Liste zurück."""
    return list(_log_buffer)


def on_new_line(callback) -> None:
    """Registriert einen Callback der bei jeder neuen stdout-Zeile aufgerufen wird."""
    _log_callbacks.append(callback)


async def start_transcription() -> bool:
    """
    Startet transcription_system.py als asyncio-Subprocess.
    Gibt False zurück wenn bereits läuft.
    """
    global _process

    if is_running():
        return False

    _log_buffer.clear()

    _process = await asyncio.create_subprocess_exec(
        str(VENV_PYTHON),
        str(SCRIPT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
        creationflags=0x00000200,  # CREATE_NEW_PROCESS_GROUP (Windows)
    )

    # Hintergrund-Task: stdout zeilenweise lesen und in Puffer schreiben
    asyncio.create_task(_read_stdout())
    return True


async def stop_transcription() -> bool:
    """
    Stoppt den laufenden Prozess.
    Sendet zuerst CTRL_BREAK_EVENT (graceful), wartet 5s, dann kill().
    Gibt False zurück wenn kein Prozess läuft.
    """
    global _process

    if not is_running():
        return False

    import subprocess
    try:
        # Windows: Gesamten Prozessbaum rigoros abtöten um Zombie-Prozesse und Handle-Locks zu vermeiden
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(_process.pid)], capture_output=True, check=False)
        
        # Sicherstellen dass der Async-Loop weiß, dass der Prozess beendet wurde
        try:
            await asyncio.wait_for(_process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
    except Exception:
        pass

    _process = None
    return True


async def _read_stdout() -> None:
    """Liest stdout des Kindprozesses zeilenweise und füllt den Log-Puffer."""
    global _process

    if _process is None or _process.stdout is None:
        return

    try:
        async for line_bytes in _process.stdout:
            try:
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
            except Exception:
                line = str(line_bytes)

            lower_line = line.lower()
            if "speak now" in lower_line or "listening..." in lower_line or "voice activity detected" in lower_line or "transcribing..." in lower_line:
                continue

            _log_buffer.append(line)

            for cb in _log_callbacks:
                try:
                    cb(line)
                except Exception:
                    pass
    except Exception:
        pass

    # Prozess ist fertig
    _process = None
