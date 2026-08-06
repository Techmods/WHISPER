# =============================================================================
# tts_process.py — Qwen TTS Subprocess (GPU)
#
# Läuft als eigenständiger Subprocess, ausschließlich auf der GPU.
# Kommuniziert mit dem Parent (voice_pipeline.py) über stdin/stdout:
#
#   stdin  (Parent → TTS):  JSON-Zeilen mit {"cmd": "speak", "text": "..."}
#                            oder {"cmd": "interrupt"} / {"cmd": "quit"}
#   stdout (TTS → Parent):  JSON-Zeilen mit {"event": "started"|"done"|"interrupted"|"error",
#                            "text": "..."}
#
# Interne Queue: eingehende Sätze werden in einer asyncio.Queue gepuffert.
# Barge-In: bei {"cmd": "interrupt"} wird laufendes Audio sofort gestoppt
# und die Queue geleert.
#
# Voraussetzungen:
#   pip install torch sounddevice numpy
#   (Qwen TTS Model muss lokal verfügbar oder per HF abrufbar sein)
#
# Starten:
#   python tts_process.py [--model <model_id>] [--voice <voice>] [--device-index <n>] [--volume <0-1>]
# =============================================================================

import argparse
import asyncio
import json
import sys
import threading
import time

import numpy as np


def _emit(event: str, text: str = "", extra: dict = None) -> None:
    """Sendet ein JSON-Event an den Parent (stdout)."""
    payload = {"event": event}
    if text:
        payload["text"] = text
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# Audio-Playback-Thread (non-blocking)
# ---------------------------------------------------------------------------

class AudioPlayer:
    """
    Spielt numpy-Audio-Arrays über sounddevice ab.
    Unterstützt sofortiges Interrupt (stop()).
    """

    def __init__(self, samplerate: int = 24000, device_index=None, volume: float = 1.0):
        self.samplerate = samplerate
        self.device_index = device_index
        self.volume = volume
        self._stop_event = threading.Event()

    def play(self, audio: np.ndarray) -> bool:
        """
        Spielt Audio synchron ab. Gibt True zurück wenn komplett abgespielt,
        False wenn per stop() unterbrochen.
        """
        import sounddevice as sd

        self._stop_event.clear()
        audio = (audio * self.volume).astype(np.float32)

        chunk_size = int(self.samplerate * 0.05)  # 50 ms Chunks
        idx = 0

        try:
            with sd.OutputStream(
                samplerate=self.samplerate,
                channels=1 if audio.ndim == 1 else audio.shape[1],
                dtype="float32",
                device=self.device_index,
            ) as stream:
                while idx < len(audio):
                    if self._stop_event.is_set():
                        return False
                    chunk = audio[idx : idx + chunk_size]
                    stream.write(chunk)
                    idx += chunk_size
        except Exception as e:
            _emit("error", extra={"detail": f"AudioPlayer: {e}"})
            return False

        return True

    def stop(self) -> None:
        """Unterbricht laufende Wiedergabe sofort."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# TTS-Modell-Wrapper
# ---------------------------------------------------------------------------

class QwenTTSModel:
    """
    Wrapper für das Qwen TTS Modell.

    Versucht zuerst das 'Qwen2.5-TTS'-Interface zu laden.
    Fällt zurück auf einen einfachen gTTS-Fallback wenn das Modell nicht
    verfügbar ist (für Entwicklung/Tests ohne GPU).
    """

    def __init__(self, model_id: str, voice: str = "default", device: str = "cuda"):
        self.model_id = model_id
        self.voice = voice
        self.device = device
        self._model = None
        self._processor = None
        self._backend = None

    def load(self) -> None:
        """Lädt das TTS-Modell (blockiert bis Modell bereit)."""
        _emit("event", extra={"event": "loading_model", "model": self.model_id})
        try:
            self._load_qwen_tts()
            self._backend = "qwen"
            _emit("event", extra={"event": "model_ready", "backend": "qwen"})
        except Exception as e:
            _emit("event", extra={"event": "model_fallback", "detail": str(e)})
            try:
                self._load_kokoro_fallback()
                self._backend = "kokoro"
                _emit("event", extra={"event": "model_ready", "backend": "kokoro"})
            except Exception as e2:
                _emit("event", extra={"event": "model_ready", "backend": "gtts_fallback", "detail": str(e2)})
                self._backend = "gtts"

    def _load_qwen_tts(self) -> None:
        """Versucht Qwen2.5-TTS zu laden (benötigt transformers + torch)."""
        import torch
        from transformers import AutoProcessor, AutoModel

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()

    def _load_kokoro_fallback(self) -> None:
        """Versucht Kokoro TTS als Fallback zu laden."""
        import kokoro
        self._model = kokoro
        self._processor = None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthetisiert Text zu Audio.
        Gibt (audio_array, samplerate) zurück.
        audio_array: float32 numpy array im Bereich [-1.0, 1.0]
        """
        if self._backend == "qwen":
            return self._synthesize_qwen(text)
        elif self._backend == "kokoro":
            return self._synthesize_kokoro(text)
        else:
            return self._synthesize_gtts_fallback(text)

    def _synthesize_qwen(self, text: str) -> tuple[np.ndarray, int]:
        import torch

        inputs = self._processor(text=text, voice=self.voice, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs)

        audio = outputs.cpu().numpy().squeeze()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0

        samplerate = getattr(self._processor, "sampling_rate", 24000)
        return audio, samplerate

    def _synthesize_kokoro(self, text: str) -> tuple[np.ndarray, int]:
        audio, samplerate = self._model.generate(text, voice=self.voice or "af_heart")
        return np.array(audio, dtype=np.float32), samplerate

    def _synthesize_gtts_fallback(self, text: str) -> tuple[np.ndarray, int]:
        """Minimaler gTTS-Fallback: schreibt MP3 in tmp, dekodiert zu numpy."""
        import io
        import tempfile

        try:
            from gtts import gTTS
            import soundfile as sf

            tts = gTTS(text=text, lang="de", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            tts.save(tmp_path)

            import subprocess
            import os

            wav_path = tmp_path.replace(".mp3", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", wav_path],
                capture_output=True,
            )
            audio, sr = sf.read(wav_path, dtype="float32")
            os.unlink(tmp_path)
            os.unlink(wav_path)
            return audio, sr
        except Exception as e:
            # Absolutes Minimum: 200 ms Stille
            _emit("error", extra={"detail": f"gTTS fallback failed: {e}"})
            return np.zeros(4800, dtype=np.float32), 24000


# ---------------------------------------------------------------------------
# Haupt-Event-Loop
# ---------------------------------------------------------------------------

async def main_loop(model: QwenTTSModel, player: AudioPlayer) -> None:
    """
    Liest stdin zeilenweise (JSON-Kommandos) und verarbeitet die Satz-Queue.
    """
    sentence_queue: asyncio.Queue[str] = asyncio.Queue()
    interrupt_event = asyncio.Event()

    async def stdin_reader() -> None:
        """Liest stdin in einem asyncio-kompatiblen Thread."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        async for raw in reader:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue

            action = cmd.get("cmd", "")
            if action == "speak":
                text = cmd.get("text", "").strip()
                if text:
                    await sentence_queue.put(text)
            elif action == "interrupt":
                # Sofort unterbrechen: Queue leeren + laufendes Audio stoppen
                while not sentence_queue.empty():
                    try:
                        sentence_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                player.stop()
                interrupt_event.set()
                _emit("interrupted")
            elif action == "quit":
                player.stop()
                while not sentence_queue.empty():
                    try:
                        sentence_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                await sentence_queue.put(None)  # Sentinel

    async def tts_worker() -> None:
        """Nimmt Sätze aus der Queue und synthetisiert + spielt sie ab."""
        while True:
            text = await sentence_queue.get()
            if text is None:
                break

            interrupt_event.clear()
            _emit("started", text)

            try:
                # Synthese (blockierend, daher in Thread-Executor)
                loop = asyncio.get_event_loop()
                audio, sr = await loop.run_in_executor(None, model.synthesize, text)

                # Playback (blockierend im Thread)
                player.samplerate = sr
                completed = await loop.run_in_executor(None, player.play, audio)

                if completed:
                    _emit("done", text)
                else:
                    _emit("interrupted", text)
            except Exception as e:
                _emit("error", text, {"detail": str(e)})

            sentence_queue.task_done()

    # Modell laden (blocking, in Executor damit Event-Loop nicht hängt)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, model.load)

    _emit("ready")

    await asyncio.gather(
        stdin_reader(),
        tts_worker(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen TTS Subprocess")
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--volume", type=float, default=0.9)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    model = QwenTTSModel(model_id=args.model, voice=args.voice, device=args.device)
    player = AudioPlayer(device_index=args.device_index, volume=args.volume)

    try:
        asyncio.run(main_loop(model, player))
    except KeyboardInterrupt:
        _emit("quit")


if __name__ == "__main__":
    main()
