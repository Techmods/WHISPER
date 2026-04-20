import sys
import os
import json
from faster_whisper import WhisperModel

from config import (
    MODEL_SIZE, COMPUTE_TYPE, DEVICE, GPU_DEVICE_INDEX,
    LANGUAGE, BEAM_SIZE, CUSTOM_VOCABULARY, INITIAL_PROMPT_EXTRA,
    TRANSCRIPTION_STYLE_PRESET
)
from transcription_system import build_initial_prompt, process_text

def run_batch(files: list[str]):
    # Prevent reloading model if not needed
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
        
    print(f"Lade Modell '{MODEL_SIZE}' in {COMPUTE_TYPE} auf {DEVICE}:{GPU_DEVICE_INDEX}...")
    try:
        model = WhisperModel(
            MODEL_SIZE, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE, 
            device_index=[GPU_DEVICE_INDEX] if isinstance(GPU_DEVICE_INDEX, int) else GPU_DEVICE_INDEX
        )
    except Exception as e:
        print(f"Modell Ladefehler: {e}")
        sys.exit(1)

    initial_prompt = build_initial_prompt(CUSTOM_VOCABULARY)

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"__BATCH_ERR__:{filepath}|Datei nicht gefunden.")
            continue
            
        print(f"Verarbeite: {os.path.basename(filepath)}")
        
        try:
            # Determine language if auto
            lang = None if LANGUAGE in ("auto", "None", "") else LANGUAGE
            
            segments, info = model.transcribe(
                filepath, 
                language=lang, 
                beam_size=BEAM_SIZE, 
                initial_prompt=initial_prompt
            )
            
            full_text = []
            for segment in segments:
                text = process_text(segment.text)
                if text.strip():
                    full_text.append(text)
            
            final_text = "\n".join(full_text)
            # Escape newlines so readline() receives the full result in one line
            safe_text = final_text.replace("\n", "<NL>")
            print(f"__BATCH_RES__:{filepath}|{safe_text}")
            
        except Exception as e:
            print(f"__BATCH_ERR__:{filepath}|{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_batch(sys.argv[1:])
    else:
        print("Keine Dateien übergeben.")
