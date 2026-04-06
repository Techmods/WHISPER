# =============================================================================
# test_gpu.py — GPU-Diagnose für das Live-Transkriptionssystem
#
# Prueft:
#   1. PyTorch CUDA-Verfuegbarkeit und GPU-Informationen
#   2. CTranslate2 CUDA-Unterstuetzung und verfuegbare Compute-Typen
#   3. faster-whisper Modell-Ladetest auf GPU (float16, Blackwell-kompatibel)
#
# Ausfuehren:
#   python test_gpu.py
# =============================================================================

import sys
import time

SECTION_WIDTH = 60


def print_header(title: str) -> None:
    print("\n" + "=" * SECTION_WIDTH)
    print(f"  {title}")
    print("=" * SECTION_WIDTH)


def print_ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def print_fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def print_info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# -----------------------------------------------------------------------------
# 1. Python-Version
# -----------------------------------------------------------------------------
print_header("1. Python-Umgebung")
print_info(f"Python {sys.version}")
print_info(f"Executable: {sys.executable}")

# -----------------------------------------------------------------------------
# 2. PyTorch und CUDA
# -----------------------------------------------------------------------------
print_header("2. PyTorch CUDA-Pruefung")
try:
    import torch

    print_ok(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda

        print_ok(f"CUDA verfuegbar: Ja")
        print_ok(f"CUDA Version (PyTorch): {cuda_version}")
        print_ok(f"GPU: {gpu_name}")
        print_ok(f"VRAM: {vram_gb:.1f} GB")
        print_info(f"Compute Capability: sm_{props.major}{props.minor}")
        print_info(f"CUDA-Geraete gesamt: {torch.cuda.device_count()}")

        # Kurzer Tensor-Test auf GPU
        try:
            t = torch.tensor([1.0, 2.0], dtype=torch.float16, device="cuda")
            _ = t * 2
            print_ok("float16 Tensor-Operation auf GPU erfolgreich")
        except Exception as e:
            print_fail(f"float16 Tensor-Test fehlgeschlagen: {e}")
    else:
        print_fail("CUDA NICHT verfuegbar — GPU-Beschleunigung nicht moeglich")
        print_info("Moegliche Ursachen: Kein NVIDIA-Treiber, CUDA nicht installiert,")
        print_info("oder PyTorch ohne CUDA-Unterstuetzung installiert.")
        print_info("Installieren: pip install torch --index-url https://download.pytorch.org/whl/cu128")

except ImportError:
    print_fail("PyTorch nicht installiert")
    print_info("Installieren: pip install torch --index-url https://download.pytorch.org/whl/cu128")

# -----------------------------------------------------------------------------
# 3. CTranslate2
# -----------------------------------------------------------------------------
print_header("3. CTranslate2 CUDA-Pruefung")
try:
    import ctranslate2

    print_ok(f"CTranslate2 Version: {ctranslate2.__version__}")

    # Unterstuetzte Compute-Typen fuer CUDA
    try:
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        print_ok(f"Unterstuetzte CUDA Compute-Typen: {', '.join(cuda_types)}")

        if "float16" in cuda_types:
            print_ok("float16 fuer CUDA unterstuetzt (Blackwell-kompatibel)")
        else:
            print_fail("float16 NICHT in unterstuetzten CUDA Typen gefunden!")

        if "int8" not in cuda_types:
            print_info("int8 nicht verfuegbar fuer CUDA — erwartet fuer Blackwell sm_120")
        else:
            print_info("int8 verfuegbar fuer CUDA")

    except Exception as e:
        print_fail(f"Compute-Typ-Abfrage fehlgeschlagen: {e}")

    # CPU Compute-Typen zur Referenz
    try:
        cpu_types = ctranslate2.get_supported_compute_types("cpu")
        print_info(f"CPU Compute-Typen: {', '.join(cpu_types)}")
    except Exception:
        pass

except ImportError:
    print_fail("CTranslate2 nicht installiert")
    print_info("Wird automatisch mit faster-whisper installiert")

# -----------------------------------------------------------------------------
# 4. faster-whisper Modell-Ladetest
# -----------------------------------------------------------------------------
print_header("4. faster-whisper Modell-Ladetest")
try:
    from faster_whisper import WhisperModel

    print_ok("faster-whisper importiert")

    # Kleines Modell fuer schnellen Test verwenden
    test_model_size = "base"
    print_info(f"Lade Testmodell '{test_model_size}' auf CUDA mit float16...")
    print_info("(Beim ersten Aufruf wird das Modell heruntergeladen — bitte warten)")

    start = time.time()
    try:
        model = WhisperModel(
            test_model_size,
            device="cuda",
            compute_type="float16",
            gpu_device_index=0,
        )
        elapsed = time.time() - start
        print_ok(f"Modell '{test_model_size}' erfolgreich auf GPU geladen ({elapsed:.1f}s)")

        # Modell wieder freigeben
        del model
        if "torch" in sys.modules:
            import torch
            torch.cuda.empty_cache()
        print_info("Modell-Speicher freigegeben")

    except Exception as e:
        print_fail(f"Modell konnte nicht geladen werden: {e}")
        print_info("Moegliche Ursachen:")
        print_info("  - Kein Internetzugang beim ersten Download")
        print_info("  - Zu wenig VRAM (base: ~0.5 GB, large-v3: ~3 GB)")
        print_info("  - compute_type='float16' nicht unterstuetzt")

except ImportError:
    print_fail("faster-whisper nicht installiert")
    print_info("Installieren: pip install faster-whisper")

# -----------------------------------------------------------------------------
# 5. RealtimeSTT Verfuegbarkeit
# -----------------------------------------------------------------------------
print_header("5. RealtimeSTT Pruefung")
try:
    from RealtimeSTT import AudioToTextRecorder
    print_ok("RealtimeSTT importiert")
    print_info("AudioToTextRecorder verfuegbar")
except ImportError:
    print_fail("RealtimeSTT nicht installiert")
    print_info("Installieren: pip install RealtimeSTT")
except Exception as e:
    print_info(f"RealtimeSTT Import-Warnung: {e}")

# -----------------------------------------------------------------------------
# Zusammenfassung
# -----------------------------------------------------------------------------
print_header("Zusammenfassung")
print_info("Pruefung abgeschlossen. Alle [OK]-Eintraege sind Voraussetzungen")
print_info("fuer das Live-Transkriptionssystem.")
print_info("")
print_info("Naechster Schritt: python transcription_system.py")
print()
