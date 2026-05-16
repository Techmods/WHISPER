# =============================================================================
# process_manager.py — asyncio Subprocess-Lifecycle für transcription_system.py
#
# start_transcription() startet transcription_system.py als Kindprozess.
# stop_transcription()  killt den Prozessbaum hart (taskkill /F /T).
# Stdout-Lines werden in einem Ringpuffer gespeichert (für ui.log()).
# [STATE:xxx]-Marker im stdout aktualisieren den Lifecycle-State (offline,
# starting, loading_model, ready, recording, stopping).
# =============================================================================

import asyncio
import collections
import os
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

# Lifecycle-State des Transkriptions-Prozesses.
# Werte: "offline" | "starting" | "loading_model" | "ready" | "recording" | "stopping"
_state: str = "offline"

# Callbacks die bei jedem State-Wechsel gefeuert werden — signatur: cb(new_state: str)
_state_callbacks: list = []


def is_running() -> bool:
    """Gibt True zurück wenn der Transkriptions-Prozess aktiv läuft."""
    return _process is not None and _process.returncode is None


def get_state() -> str:
    """Aktueller Lifecycle-State (siehe Werte oben)."""
    return _state


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


def on_state_change(callback) -> None:
    """Registriert einen Callback der bei jedem State-Wechsel aufgerufen wird."""
    _state_callbacks.append(callback)


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
    global _last_log_line
    _last_log_line = ""
    _set_state("starting")

    # -u + PYTHONUNBUFFERED: stdout des Kindprozesses unbuffered, damit
    # die UI sofort "Initialisiere Modell…" sieht statt 10-30 s Stille.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

    _process = await asyncio.create_subprocess_exec(
        str(VENV_PYTHON),
        "-u",
        str(SCRIPT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
        env=env,
        creationflags=0x00000200,  # CREATE_NEW_PROCESS_GROUP (Windows)
    )

    asyncio.create_task(_read_stdout())
    return True


async def stop_transcription() -> bool:
    """
    Stoppt den laufenden Prozess hart via taskkill /F /T (Windows).
    Wartet bis zu 3 s auf sauberes Ende.
    """
    global _process

    if not is_running():
        return False

    _set_state("stopping")

    import subprocess
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(_process.pid)],
            capture_output=True,
            check=False,
        )
        try:
            await asyncio.wait_for(_process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass
    except Exception:
        pass

    _process = None
    _set_state("offline")
    return True


# Spam-Zeilen die nicht in den UI-Log sollen (häufige RealtimeSTT-Internals).
# Bewusst klein gehalten: lieber etwas mehr Log als gar keinen.
_SPAM_PATTERNS = (
    "voice activity detected",
)

# Zuletzt geloggte Zeile zur Dedupe-Erkennung.
_last_log_line: str = ""


def _handle_state_marker(line: str) -> bool:
    """Wenn line ein [STATE:xxx]-Marker ist, State setzen und True zurückgeben."""
    stripped = line.strip()
    if not stripped.startswith("[STATE:") or not stripped.endswith("]"):
        return False
    new_state = stripped[len("[STATE:"):-1].strip()
    if new_state:
        _set_state(new_state)
    return True


async def _read_stdout() -> None:
    """Liest stdout des Kindprozesses zeilenweise und füllt Log + State."""
    global _process, _last_log_line

    if _process is None or _process.stdout is None:
        return

    try:
        async for line_bytes in _process.stdout:
            try:
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
            except Exception:
                line = str(line_bytes)

            if _handle_state_marker(line):
                continue

            lower_line = line.lower()
            if any(p in lower_line for p in _SPAM_PATTERNS):
                continue

            # Transkript-Ergebnisse (__TRANSCRIPT__:...) dürfen nicht dedupiert
            # werden — zwei identische Aussagen sind legitim.
            is_transcript = line.startswith("__TRANSCRIPT__:")
            if not is_transcript and line == _last_log_line:
                continue
            _last_log_line = line

            _log_buffer.append(line)

            for cb in _log_callbacks:
                try:
                    cb(line)
                except Exception:
                    pass
    except Exception:
        pass

    _process = None
    _last_log_line = ""
    _set_state("offline")
