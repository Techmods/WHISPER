# =============================================================================
# config_rw.py — AST-basierter Lese/Schreib-Zugriff auf config.py
#
# Liest Werte via importlib (sicher, isoliert).
# Schreibt Werte via AST-Positions-Splice (erhält alle Kommentare).
# Atomares Schreiben: .py.tmp -> os.replace()
# =============================================================================

import ast
import os
import importlib.util
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.py"

# Alle Keys die die UI lesen/schreiben darf
UI_KEYS = {
    "MODEL_SIZE",
    "COMPUTE_TYPE",
    "DEVICE",
    "GPU_DEVICE_INDEX",
    "LANGUAGE",
    "BEAM_SIZE",
    "INITIAL_PROMPT_EXTRA",
    "CUSTOM_VOCABULARY",
    "KEYWORD_EXPANSIONS",
    "CORRECTIONS",
    "INPUT_DEVICE_INDEX",
    "VAD_ENABLED",
    "SILERO_SENSITIVITY",
    "MIN_LENGTH_OF_RECORDING",
    "PRE_RECORDING_BUFFER_DURATION",
    "TYPE_INTO_CURSOR",
    "OUTPUT_FILE",
}


def read_config() -> dict:
    """
    Liest config.py in einen isolierten Namespace und gibt die bekannten Keys zurück.
    Sicher für reines Lesen — kein Schreiben passiert hier.
    """
    spec = importlib.util.spec_from_file_location("_whisper_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = {}
    for key in UI_KEYS:
        if hasattr(module, key):
            result[key] = getattr(module, key)
    return result


def write_config(updates: dict) -> None:
    """
    Schreibt geänderte Werte zurück in config.py per AST-Positions-Splice.
    Alle Kommentare und unveränderte Zeilen bleiben erhalten.
    Atomares Schreiben via temporäre Datei + os.replace().

    Args:
        updates: Dict mit Key -> neuer Wert (nur UI_KEYS werden akzeptiert)
    """
    # Nur erlaubte Keys
    updates = {k: v for k, v in updates.items() if k in UI_KEYS}
    if not updates:
        return

    raw = CONFIG_PATH.read_text(encoding="utf-8")

    # CRLF normalisieren für AST (Windows-Zeilenenden)
    crlf = "\r\n" in raw
    source = raw.replace("\r\n", "\n")

    tree = ast.parse(source)

    # Alle Top-Level Assign-Nodes mit ihren Wert-Spans sammeln
    # Reihenfolge: von hinten nach vorne, damit Offsets beim Ersetzen stabil bleiben
    replacements = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in updates:
            continue

        val_node = node.value
        # Zeilenbasierte Positionen in Zeichenoffsets umrechnen
        lines = source.splitlines(keepends=True)
        start_offset = _line_col_to_offset(lines, val_node.lineno, val_node.col_offset)
        end_offset = _line_col_to_offset(lines, val_node.end_lineno, val_node.end_col_offset)

        new_repr = _to_repr(updates[target.id])
        replacements.append((start_offset, end_offset, new_repr))

    if not replacements:
        return

    # Von hinten nach vorne ersetzen (Offsets bleiben stabil)
    replacements.sort(key=lambda x: x[0], reverse=True)
    for start, end, new_repr in replacements:
        source = source[:start] + new_repr + source[end:]

    # CRLF wiederherstellen wenn original CRLF
    if crlf:
        source = source.replace("\n", "\r\n")

    tmp_path = CONFIG_PATH.with_suffix(".py.tmp")
    tmp_path.write_text(source, encoding="utf-8")
    os.replace(tmp_path, CONFIG_PATH)


def _line_col_to_offset(lines: list, lineno: int, col_offset: int) -> int:
    """Konvertiert 1-basierte Zeile + 0-basierte Spalte in absoluten Zeichenoffset."""
    offset = sum(len(l) for l in lines[:lineno - 1])
    return offset + col_offset


def _to_repr(value) -> str:
    """
    Gibt eine gültige Python-Literal-Darstellung zurück.
    Dicts und Listen werden einzeilig formatiert.
    Strings mit Backslashes werden als r-Strings dargestellt wenn möglich.
    """
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return "None"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = ",\n    ".join(repr(v) for v in value)
        return f"[\n    {items},\n]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ",\n    ".join(f"{repr(k)}: {repr(v)}" for k, v in value.items())
        return f"{{\n    {items},\n}}"
    return repr(value)