# =============================================================================
# Alternative Prompt-Strategien fuer Whisper, um die Erkennung von Fachbegriffen zu verbessern.
# =============================================================================

def build_initial_prompt(vocabulary: list) -> str:
    """
    Erstellt den initial_prompt fuer Whisper aus der Custom-Vocabulary-Liste.

    Der initial_prompt gibt Whisper Kontext-Hinweise fuer die Transkription.
    Er sollte die wichtigsten Fachbegriffe als repraesentativen Satz enthalten.

    Args:
        vocabulary: Liste der Fachbegriffe aus CUSTOM_VOCABULARY.

    Returns:
        Kommaseparierter String der Fachbegriffe als Prompt-Text.
    """
    if not vocabulary:
        return ""
    # Fachbegriffe als natuerllichen Satz formulieren verbessert die Erkennung
    terms = ", ".join(vocabulary)
    return f"Fachbegriffe: {terms}."