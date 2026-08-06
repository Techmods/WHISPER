# =============================================================================
# llm_client.py — OpenRouter Streaming-Client mit Satz-Chunking
#
# Stellt eine asynchrone Funktion bereit, die den Token-Stream der
# OpenRouter API (SSE) empfängt, Sätze nach Satzzeichen puffert und
# satzweise über einen Callback übergibt.
#
# Unterstützt Interrupt: Bei gesetztem asyncio.Event wird der Stream
# sofort abgebrochen.
#
# Voraussetzungen:
#   pip install httpx
# =============================================================================

from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncIterator, Callable, Awaitable

# Satzende-Muster: Punkt, Ausrufezeichen, Fragezeichen (gefolgt von Leerzeichen
# oder String-Ende), sowie Doppelpunkt am Satzende.
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+|(?<=[.!?])$')

# Minimale Zeichenanzahl bevor ein Chunk an TTS übergeben wird
_MIN_CHUNK_LEN = 8


def _split_into_sentences(text: str) -> tuple[list[str], str]:
    """
    Teilt text in abgeschlossene Sätze und einen verbleibenden Puffer.
    Gibt (fertige_saetze, rest) zurück.
    """
    parts = _SENTENCE_END_RE.split(text)
    if len(parts) <= 1:
        return [], text

    # Der letzte Teil ist noch nicht abgeschlossen (kein Satzende gesehen)
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    return complete, remainder


async def stream_llm_response(
    api_key: str,
    model: str,
    messages: list[dict],
    on_sentence: Callable[[str], Awaitable[None]],
    interrupt_event: asyncio.Event | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Streamt eine LLM-Antwort von der OpenRouter API (SSE).

    Sätze werden satzweise (nach `.`, `!`, `?`) an `on_sentence` übergeben,
    sobald ein Satzende erkannt wird — ohne auf die komplette Antwort zu warten.

    Args:
        api_key:         OpenRouter API-Schlüssel.
        model:           Modell-ID (z.B. "openai/gpt-4o-mini").
        messages:        Chat-History im OpenAI-Format.
        on_sentence:     Async Callback — wird für jeden fertigen Satz aufgerufen.
        interrupt_event: Wenn gesetzt und Event gefeuert, Stream sofort abbrechen.
        base_url:        OpenRouter API-URL.
        temperature:     Sampling-Temperatur.
        max_tokens:      Maximale Antwortlänge in Tokens.

    Returns:
        Vollständiger generierter Text (für Kontext-History).
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx fehlt: pip install httpx") from e

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Techmods/WHISPER",
        "X-Title": "Whisper Voice Pipeline",
    }

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    buffer = ""
    full_text = ""

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()

            async for raw_line in response.aiter_lines():
                # Interrupt-Check
                if interrupt_event is not None and interrupt_event.is_set():
                    break

                if not raw_line.startswith("data: "):
                    continue

                data = raw_line[len("data: "):]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if not delta:
                    continue

                buffer += delta
                full_text += delta

                # Satz-Chunking: fertige Sätze aus dem Puffer entnehmen
                sentences, buffer = _split_into_sentences(buffer)
                for sentence in sentences:
                    if len(sentence) >= _MIN_CHUNK_LEN:
                        await on_sentence(sentence)

    # Verbleibenden Puffer als letzten Chunk ausgeben
    remainder = buffer.strip()
    if remainder and len(remainder) >= _MIN_CHUNK_LEN:
        await on_sentence(remainder)

    return full_text


async def build_messages(
    system_prompt: str,
    history: list[dict],
    user_text: str,
) -> list[dict]:
    """
    Erstellt die Messages-Liste für die OpenRouter API.
    Hängt die aktuelle User-Aussage an die bestehende History an.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages
