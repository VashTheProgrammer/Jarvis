import os
import time
import subprocess
import platform

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map

# =============================================================================
#                               CONFIGURAZIONE
# =============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Nome del modello su HuggingFace
MODEL_DIR = os.path.expanduser("~/models")         # Cartella locale per il caching dei pesi
PROMPT = "Qual è la capitale dell'Italia?"         # Prompt di esempio
RAM_LIMIT_GB = 12                                  # Soglia di memoria: se superata -> MemoryError
BATCH_SIZE = 1                                     # Non usato esplicitamente, ma presente per eventuali estensioni

# Alcune variabili globali (evitiamo, se possibile, di abusarne)
DEVICE = None           # Viene impostato in system_info()
MAX_RAM_USED_GB = 0.0   # Per tracciare il picco di RAM usata

# =============================================================================
#                          FUNZIONI DI LOG E UTILS
# =============================================================================

def log(message: str) -> None:
    """
    Stampa un messaggio con timestamp (formato [YYYY-MM-DD HH:MM:SS]).
    """
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] {message}")


# =============================================================================
#                      FUNZIONI PER MISURAZIONE DELLA MEMORIA
# =============================================================================

def check_memory_usage(limit_gb: float = RAM_LIMIT_GB) -> tuple[float, float, float]:
    """
    Legge l'uso della RAM dal sistema tramite psutil.
    - Aggiorna la variabile globale MAX_RAM_USED_GB se l'uso attuale supera il record precedente.
    - Se l'uso di RAM supera 'limit_gb', lancia MemoryError.
    - Ritorna (used_gb, available_gb, total_gb).
    """
    global MAX_RAM_USED_GB

    mem_info = psutil.virtual_memory()
    used_gb = mem_info.used / (1024**3)
    available_gb = mem_info.available / (1024**3)
    total_gb = mem_info.total / (1024**3)

    # Aggiorna picco massimo
    if used_gb > MAX_RAM_USED_GB:
        MAX_RAM_USED_GB = used_gb

    # Controlla la soglia di memoria
    if used_gb > limit_gb:
        raise MemoryError(
            f"Uso di RAM {used_gb:.2f} GB > soglia di {limit_gb} GB!"
        )

    return used_gb, available_gb, total_gb

def get_max_memory_usage() -> float:
    """
    Restituisce il picco massimo di RAM (in GB) usata durante l'esecuzione.
    """
    return MAX_RAM_USED_GB

# =============================================================================
#                          FUNZIONI DI SISTEMA (macOS)
# =============================================================================

def get_macOS_version() -> str:
    """
    Restituisce la versione di macOS e il relativo 'nome marketing' (es. Sonoma, Ventura, ecc.).
    """
    try:
        version = subprocess.check_output(["sw_vers", "-productVersion"]).decode("utf-8").strip()
        major_version = int(version.split(".")[0])
        
        macos_names = {
            15: "Sequoia",
            14: "Sonoma",
            13: "Ventura",
            12: "Monterey",
            11: "Big Sur",
            10: "Catalina"
        }
        return f"macOS {version} ({macos_names.get(major_version, 'Sconosciuto')})"
    except Exception:
        return "macOS (Versione non rilevata)"

def get_gpu_name() -> str:
    """
    Recupera il nome della GPU OpenCL (utile per macOS con Apple Silicon).
    """
    try:
        result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
        lines = result.stdout.split("\n")
        for line in lines:
            if "Chipset Model:" in line:
                return line.split(":")[1].strip()
    except Exception:
        return "GPU non rilevata"
    return "N/A"

def get_cpu_info() -> dict:
    """
    Restituisce informazioni dettagliate sulla CPU per macOS.
    Tenta un approccio combinato: prima psutil, poi sysctl, infine system_profiler.
    """
    info = {
        "Architettura": platform.machine(),
        "Core Fisici": "N/A",
        "Core Logici": "N/A",
        "Performance Core": "N/A",
        "Efficiency Core": "N/A",
        "Frequenza (MHz)": "N/A"
    }

    # Prova psutil
    try:
        freq = psutil.cpu_freq()
        if freq:
            info["Frequenza (MHz)"] = round(freq.current, 2)
    except:
        pass

    try:
        info["Core Fisici"] = psutil.cpu_count(logical=False)
        info["Core Logici"] = psutil.cpu_count(logical=True)
    except:
        pass

    # Fallback con sysctl se la frequenza non è stata letta correttamente
    if info["Frequenza (MHz)"] in ["N/A", 0.0]:
        try:
            cpu_speed_hz = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency"], capture_output=True, text=True
            ).stdout.strip()
            if cpu_speed_hz.isdigit():
                info["Frequenza (MHz)"] = round(int(cpu_speed_hz) / 1e6, 2)
        except:
            # Ultimo tentativo con system_profiler (macchine Intel)
            try:
                cpu_data = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
                for line in cpu_data.split("\n"):
                    if "Processor Speed" in line:
                        # es. "Processor Speed: 3.2 GHz"
                        freq_str = line.split(":")[1].strip()
                        info["Frequenza (MHz)"] = freq_str
                        break
            except:
                pass

    # Legge performance/efficiency core (Apple Silicon)
    try:
        perf_cores = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout.strip()
        eff_cores = subprocess.run(["sysctl", "-n", "hw.perflevel1.physicalcpu"], capture_output=True, text=True).stdout.strip()

        if perf_cores.isdigit():
            info["Performance Core"] = perf_cores
        if eff_cores.isdigit():
            info["Efficiency Core"] = eff_cores
    except:
        pass

    return info

def get_system_info() -> None:
    """
    Stampa informazioni di sistema dettagliate (macOS): versione, CPU, GPU, RAM, etc.
    Inizializza la variabile globale DEVICE in base alla disponibilità MPS.
    """
    global DEVICE

    # OS e GPU
    os_info = get_macOS_version()
    gpu_name = get_gpu_name()
    cpu_info = get_cpu_info()

    # Verifica RAM (e fa un primo check soglia)
    used_gb, available_gb, total_gb = check_memory_usage(RAM_LIMIT_GB)

    # Stampa
    log(f"Sistema Operativo: {os_info}")
    log(f"CPU: {platform.processor()} ({cpu_info['Architettura']})")
    log(f" - Core Fisici: {cpu_info['Core Fisici']} | Core Logici: {cpu_info['Core Logici']}")
    log(f" - Performance Core: {cpu_info['Performance Core']} | Efficiency Core: {cpu_info['Efficiency Core']}")
    log(f" - Frequenza: {cpu_info['Frequenza (MHz)']} MHz")
    log(f"GPU: {gpu_name}")
    log(f"RAM Totale: {total_gb:.2f} GB | Disponibile: {available_gb:.2f} GB")
    
    # MPS check
    mps_available = torch.backends.mps.is_available()
    DEVICE = torch.device("mps") if mps_available else torch.device("cpu")
    log(f"Accelerazione MPS: {'Sì' if mps_available else 'No'} | Dispositivo: {DEVICE}")


# =============================================================================
#                     FUNZIONI PER MODELLO E INFERENZA
# =============================================================================

def load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer, float]:
    """
    Carica il modello e il tokenizer da HuggingFace, 
    usando i pesi in float16 e device_map="auto".
    Restituisce (modello, tokenizer, tempo_caricamento).
    """
    log("Inizio caricamento modello...")
    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token

    load_time = round(time.time() - start_time, 2)

    # Controllo RAM dopo il caricamento
    used_gb, available_gb, total_gb = check_memory_usage(RAM_LIMIT_GB)
    log(f"Modello caricato in {load_time} sec (RAM in uso: {used_gb:.2f}/{total_gb:.2f} GB)")

    return model, tokenizer, load_time

def run_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> tuple[str, float]:
    """
    Esegue inferenza con il modello e ritorna (testo_generato, tempo_inferenza).
    """
    log("Avvio inferenza...")
    start_time = time.time()

    # Prepara input
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Possiamo anche misurare la CPU usage prima/dopo l'inferenza
    cpu_usage_before = psutil.cpu_percent(interval=None)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    cpu_usage_after = psutil.cpu_percent(interval=None)

    elapsed = round(time.time() - start_time, 2)
    result_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Controllo RAM dopo inferenza
    used_gb, available_gb, total_gb = check_memory_usage(RAM_LIMIT_GB)

    log(f"Inferenza completata in {elapsed} sec (RAM in uso: {used_gb:.2f}/{total_gb:.2f} GB)")
    log(f"CPU usage (prima -> dopo inferenza): {cpu_usage_before:.1f}% -> {cpu_usage_after:.1f}%")

    return result_text, elapsed

# =============================================================================
#                          MAIN SCRIPT
# =============================================================================

def main():
    """
    Funzione principale: mostra info di sistema, carica modello,
    esegue inferenza, e stampa un resoconto prestazioni.
    """
    # Crea cartella modelli, se non esiste
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        # 1) Info di sistema
        get_system_info()
        
        # 2) Caricamento modello
        model, tokenizer, load_time = load_model()
        
        # 3) Esecuzione inferenza
        response, inference_time = run_inference(model, tokenizer, PROMPT)

        # 4) Picco massimo di RAM usata
        peak_ram_gb = get_max_memory_usage()

        # 5) Report finale
        log("===== RESOCONTO PRESTAZIONI =====")
        log(f"Tempo Caricamento Modello: {load_time} sec")
        log(f"Tempo Inferenza: {inference_time} sec")
        log(f"Risultato Inferenza: {response}")
        log(f"Memoria Massima Utilizzata: {peak_ram_gb:.2f} GB")

    except MemoryError as me:
        log(f"ERRORE DI MEMORIA: {me}")
    except Exception as e:
        log(f"ERRORE: {e}")

if __name__ == "__main__":
    main()
