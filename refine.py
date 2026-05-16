# =============================================================================
# refine.py — LLM-Post-Processing über LM Studio (OpenAI-kompatibel)
#
# Eine Funktion `polish()` kombiniert in einem einzigen LLM-Call:
#   - Backtrack/Self-Correction-Auflösung
#   - Filler-Wort-Entfernung
#   - Übersetzung in eine Zielsprache
#
# Wird von transcription_system.process_text() als letzter Schritt nach
# Korrekturen + Keyword-Expansionen aufgerufen.
#
# Fehler werden als RefineError gehoben; der Aufrufer entscheidet, ob er
# auf rohen Text zurückfällt.
# =============================================================================

import json
import urllib.request
import urllib.error

LM_STUDIO_ENDPOINT_DEFAULT = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODELS_URL_DEFAULT = "http://localhost:1234/v1/models"

# ISO 639-1 → menschlich lesbarer Sprachname für den Prompt.
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "tr": "Turkish",
}


class RefineError(Exception):
    """Refine-Pass ist fehlgeschlagen. Aufrufer fällt auf Rohtext zurück."""


def _models_url_from_chat_endpoint(endpoint: str) -> str:
    """
    Leitet die /v1/models-URL aus der Chat-Endpoint-URL ab.
    Bspw. http://host:1234/v1/chat/completions -> http://host:1234/v1/models
    """
    marker = "/chat/completions"
    if endpoint.endswith(marker):
        return endpoint[: -len(marker)] + "/models"
    if "/v1/" in endpoint:
        return endpoint.split("/v1/")[0] + "/v1/models"
    return LM_STUDIO_MODELS_URL_DEFAULT


def list_models(endpoint: str = LM_STUDIO_ENDPOINT_DEFAULT, timeout: float = 3.0) -> list[str]:
    """
    Holt die in LM Studio geladenen Modelle. Bei Fehler: leere Liste
    (kein Exception — die Funktion ist für UI-Dropdown-Befüllung gedacht
    und soll nicht den App-Start blockieren).
    """
    url = _models_url_from_chat_endpoint(endpoint)
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in data.get("data", []) if "id" in m]
    except Exception:
        return []


def _build_system_prompt(translate: bool, target_language: str, strip_fillers: bool, backtrack: bool) -> str:
    parts = [
        "You are a precise polisher of raw speech-to-text transcripts.",
        "Apply the following operations IN ORDER, then output the result.",
        "",
    ]
    step = 1
    if backtrack:
        parts.append(
            f"{step}. RESOLVE SELF-CORRECTIONS: when the speaker corrects themselves "
            f"(\"no wait\", \"actually\", \"I mean\", \"ne, Freitag\", \"nein\"), "
            f"keep ONLY the corrected version. Remove the original mistake completely."
        )
        step += 1
    if strip_fillers:
        parts.append(
            f"{step}. REMOVE FILLER WORDS: ähm, äh, also, halt, naja, you know, like, "
            f"um, uh, and similar discourse particles. Keep the natural flow."
        )
        step += 1
    if translate:
        lang_name = LANGUAGE_NAMES.get(target_language, target_language)
        parts.append(
            f"{step}. TRANSLATE the result into {lang_name}. Use natural, idiomatic "
            f"phrasing. Keep proper nouns, code identifiers, and numbers exactly as they appear."
        )
        step += 1
    parts.append("")
    parts.append("OUTPUT ONLY the polished text. No explanations, no quotation marks, no markdown.")
    return "\n".join(parts)


def polish(
    text: str,
    *,
    endpoint: str = LM_STUDIO_ENDPOINT_DEFAULT,
    model: str,
    translate: bool = False,
    target_language: str = "en",
    strip_fillers: bool = False,
    backtrack: bool = False,
    timeout: float = 15.0,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    min_words: int = 3,
) -> str:
    """
    Polish raw transcript via LM Studio. Genau ein HTTP-Call.

    Returns:
        Polierter Text. Wenn KEIN Modus aktiv ist oder Input zu kurz: text unverändert.

    Raises:
        RefineError bei Netz-/Server-/Parsing-Fehlern.
    """
    if not (translate or strip_fillers or backtrack):
        return text

    stripped = text.strip()
    if len(stripped.split()) < min_words:
        return text  # nicht lohnenswert — Rohtext zurück

    system_prompt = _build_system_prompt(translate, target_language, strip_fillers, backtrack)

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": stripped},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RefineError(f"HTTP {e.code} from LM Studio: {body_text or e.reason}") from e
    except urllib.error.URLError as e:
        raise RefineError(f"LM Studio nicht erreichbar ({endpoint}): {e.reason}") from e
    except Exception as e:
        raise RefineError(f"Refine-Request fehlgeschlagen: {e}") from e

    try:
        data = json.loads(raw)
        polished = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RefineError(f"Refine-Response unverständlich: {e}") from e

    if not polished:
        raise RefineError("LM Studio lieferte leeren Output zurück.")

    return polished
