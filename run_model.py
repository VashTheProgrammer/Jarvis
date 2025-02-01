import os

os.environ["HF_HOME"] = "/Users/adrianocostanzo/IA/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/Users/adrianocostanzo/IA/Datasets"

import time
import psutil
import torch
import gc
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================
# Configurazioni (modifica direttamente qui)
# ============================================

# 📌 Scegli il modello da eseguire (Modifica questa variabile per cambiare modello)
MODEL_NAME = "microsoft/phi-2"  
# Sostituiscilo con uno tra:
# "mistralai/Mistral-7B-Instruct-v0.1"
# "deepseek-ai/deepseek-r1-7b"
# "meta-llama/Llama-2-7b-chat-hf"
# "microsoft/phi-2"

# 📌 Imposta la directory in cui salvare i modelli
MODEL_DIR = os.path.expanduser("/Users/adrianocostanzo/IA/Models")  # Modifica per la tua directory preferita

TOKEN_COUNT = 100  # Numero di token da generare per il test

# 📌 Prompt di input tradotto in inglese e leggermente modificato per stimolare una risposta
PROMPT = "Hello, who are you? Who am I speaking with? Give me a direct and concise answer. Answer:"

# ============================================

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Forza l'uso della CPU (ignora CUDA e MPS)
DEVICE = "cpu"
logging.info(f"Dispositivo forzato: {DEVICE}")

def get_system_usage():
    """Restituisce l'utilizzo attuale di CPU e RAM."""
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    return cpu_usage, ram_usage

def clear_memory():
    """Libera la memoria: esegue gc.collect() e svuota la cache se disponibile."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(model_name, model_dir, device):
    """Scarica e carica il modello e il tokenizer specificati."""
    logging.info(f"📥 [1/3] Downloading & Loading model: {model_name}...")
    os.makedirs(model_dir, exist_ok=True)
    clear_memory()
    
    cpu_before, ram_before = get_system_usage()
    logging.info(f"  📊 CPU Usage: {cpu_before}% | RAM Usage: {ram_before}% (before loading)")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=model_dir,
            device_map=device  # Usa il dispositivo configurato ("cpu")
        )
        # Imposta il modello in modalità evaluation per l'inferenza
        model.eval()
    except Exception as e:
        logging.error("Errore durante il caricamento del modello:", exc_info=True)
        raise e

    cpu_after, ram_after = get_system_usage()
    logging.info(f"  ✅ Model loaded! CPU Usage: {cpu_after}% | RAM Usage: {ram_after}% (after loading)\n")
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens):
    """
    Genera testo a partire dal prompt, misurando il tempo di inferenza.
    Include:
      - Un passaggio di warm-up per "riscaldare" il modello.
      - L'uso di torch.inference_mode() per disabilitare il calcolo dei gradienti.
      - Parametri di generazione per ottenere una risposta varia.
      - Ottimizzazioni per liberare la memoria dopo la generazione.
    """
    logging.info(f"🚀 [2/3] Running inference for {max_new_tokens} tokens...")
    
    # Preparazione degli input e invio al dispositivo configurato
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Warm-up: esegue una chiamata iniziale per "riscaldare" il modello
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5)
    
    # Misurazione dell'inferenza vera e propria con sampling per evitare l'echo del prompt
    start_time = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    end_time = time.perf_counter()
    
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    inference_time = end_time - start_time
    tokens_per_second = max_new_tokens / inference_time if inference_time > 0 else 0
    logging.info(f"✅ Inference complete! Time taken: {inference_time:.2f} sec | Tokens/sec: {tokens_per_second:.2f}")
    
    # Ottimizzazione della memoria: elimina variabili temporanee e richiede la garbage collection
    del output, inputs
    gc.collect()
    
    return response_text, inference_time, tokens_per_second

def main():
    logging.info("🖥️ Starting AI Model Execution Benchmark...")
    clear_memory()
    
    # Caricamento del modello e del tokenizer
    model, tokenizer = load_model(MODEL_NAME, MODEL_DIR, DEVICE)
    
    # Generazione del testo e misurazione dell'inferenza
    response, inference_time, tokens_per_second = generate_text(model, tokenizer, PROMPT, TOKEN_COUNT)
    
    cpu_final, ram_final = get_system_usage()
    logging.info(f"📊 Final System Usage: CPU {cpu_final}% | RAM {ram_final}%")
    
    # Visualizzazione dei risultati con stampa chiara del prompt e della risposta del modello
    print("\n📌 [3/3] Risultati dell'Esecuzione del Modello AI:")
    print("----------------------------------------------------------")
    print("📝 Prompt inserito:")
    print(PROMPT)
    print("\n🤖 Risposta del modello:")
    print(response)
    print("----------------------------------------------------------")
    
    logging.info("✅ AI Model Execution Complete!\n")

if __name__ == "__main__":
    main()
