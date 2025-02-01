import os

os.environ["HF_HOME"] = "/Users/adrianocostanzo/IA/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/Users/adrianocostanzo/IA/Datasets"

import time
import psutil
import torch
import gc
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# ============================================
# CONFIGURAZIONE
# ============================================

# Imposta la modalità di esecuzione:
#   - "inference": genera una risposta a partire da un prompt
#   - "train": esegue il fine tuning del modello su un dataset
MODE = "inference"  # Scegli "inference" oppure "train"

# ============================================
# Impostazioni del modello
# ============================================
# Se stai eseguendo inference:
#   - Se MODEL_PATH è diverso da stringa vuota, lo script caricherà il modello fine tuned da quel percorso.
#   - Altrimenti, caricherà il modello pre-addestrato indicato in MODEL_NAME (scaricandolo da Hugging Face).
MODEL_NAME = "microsoft/phi-2"      # Nome del modello pre-addestrato (usato se MODEL_PATH è vuoto)
#MODEL_PATH = "fine_tuned_jarvis"  # Per inference: se hai un modello fine tuned, specifica qui il percorso (es. "/Users/tuoutente/IA/Models/fine_tuned")
MODEL_PATH = ""

# Directory per il caching e il salvataggio dei modelli
MODEL_DIR = os.path.expanduser("/Users/adrianocostanzo/IA/Models")

# ============================================
# Impostazioni per l'inference
# ============================================
PROMPT = "Ciao, chi sei ? a cosa servi ? chi ti ha creato ?"
MAX_TOKENS = 100

# ============================================
# Impostazioni per il fine tuning (training)
# ============================================
TRAIN_FILE = os.path.expanduser("/Users/adrianocostanzo/IA/Jarvis/training_data.jsonl")  # File JSONL contenente il dataset di training
OUTPUT_DIR = os.path.join(MODEL_DIR, "fine_tuned_jarvis")           # Directory in cui salvare il modello fine tuned
EPOCHS = 1
BATCH_SIZE = 1

# ============================================
# Impostazioni per il dispositivo
# ============================================
DEVICE = "cpu"  # Usa "cuda" se disponi di una GPU

# ============================================
# Configurazione del logging
# ============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================
# FUNZIONI UTILI
# ============================================

def get_system_usage():
    """Restituisce l'utilizzo attuale di CPU e RAM."""
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    return cpu_usage, ram_usage

def clear_memory():
    """Libera la memoria eseguendo la garbage collection e svuotando la cache di CUDA (se disponibile)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================
# FUNZIONE DI CARICAMENTO DEL MODELLO
# ============================================
def load_model(model_name, model_path, model_dir, device):
    """
    Carica il modello e il tokenizer.
    Se model_path è fornito (non vuoto), carica il modello fine tuned da quel percorso.
    Altrimenti, carica il modello pre-addestrato usando model_name (scaricandolo da Hugging Face).
    """
    clear_memory()
    cpu_before, ram_before = get_system_usage()
    logging.info(f"📊 Uso CPU: {cpu_before}% | Uso RAM: {ram_before}% (prima del caricamento)")
    
    try:
        if model_path:
            logging.info(f"📥 Caricamento del modello fine tuned da: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=model_dir, device_map=device)
        else:
            logging.info(f"📥 Caricamento del modello pre-addestrato: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir, device_map=device)
        
        # Verifica se il tokenizer ha un token di padding; in caso contrario, lo imposta
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logging.info("Imposto il token di padding uguale al token EOS.")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logging.info("Aggiungo un token di padding: [PAD].")
        
        model.eval()  # Imposta il modello in modalità evaluation
    except Exception as e:
        logging.error("Errore durante il caricamento del modello:", exc_info=True)
        raise e
    
    cpu_after, ram_after = get_system_usage()
    logging.info(f"✅ Modello caricato! Uso CPU: {cpu_after}% | Uso RAM: {ram_after}% (dopo il caricamento)")
    return model, tokenizer


# ============================================
# FUNZIONE PER L'INFERENCE
# ============================================
def generate_text(model, tokenizer, prompt, max_new_tokens, device):
    """
    Genera testo a partire dal prompt.
    Esegue un warm-up e poi la generazione disabilitando il calcolo dei gradienti.
    """
    logging.info(f"🚀 Esecuzione inference per {max_new_tokens} token...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warm-up per riscaldare il modello
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5)
    
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
    logging.info(f"✅ Inference completata in {inference_time:.2f} sec | Tokens/sec: {tokens_per_second:.2f}")
    
    del output, inputs
    gc.collect()
    
    return response_text, inference_time, tokens_per_second

# ============================================
# FUNZIONE PER IL FINE TUNING (TRAINING)
# ============================================
def fine_tune_model(model, tokenizer, train_file, output_dir, num_train_epochs=1, per_device_train_batch_size=1):
    """
    Esegue il fine tuning del modello su un dataset in formato JSONL.
    Il file JSONL deve contenere, per ogni riga, un oggetto JSON con almeno il campo "text".
    """
    logging.info(f"🚀 Caricamento dataset di training da: {train_file}")
    dataset = load_dataset('json', data_files={'train': train_file})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length"
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        report_to="none",
        no_cuda=True  # Forza l'uso della CPU, anche se sono presenti GPU/MPS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    logging.info("🚀 Avvio training...")
    trainer.train()
    logging.info("✅ Training completato!")
    
    trainer.save_model(output_dir)
    logging.info(f"✅ Modello fine tuned salvato in: {output_dir}")
    return model

# ============================================
# FUNZIONE PRINCIPALE
# ============================================
def main():
    """
    Funzione principale che gestisce la modalità di esecuzione.
    Se MODE è "inference": esegue l'inference sul modello specificato.
    Se MODE è "train": esegue il fine tuning sul dataset di training e salva il modello.
    """
    if MODE == "inference":
        # Per l'inference, se MODEL_PATH è specificato verrà usato, altrimenti viene usato MODEL_NAME
        model, tokenizer = load_model(MODEL_NAME, MODEL_PATH, MODEL_DIR, DEVICE)
        response, inference_time, tokens_per_second = generate_text(model, tokenizer, PROMPT, MAX_TOKENS, DEVICE)
        
        print("\n--- RISULTATI INFERENCE ---")
        print("Prompt:")
        print(PROMPT)
        print("\nRisposta:")
        print(response)
        print(f"\nTempo di inference: {inference_time:.2f} sec | Tokens/sec: {tokens_per_second:.2f}")
    elif MODE == "train":
        # Per il training si carica il modello pre-addestrato indicato in MODEL_NAME e si esegue il fine tuning
        model, tokenizer = load_model(MODEL_NAME, "", MODEL_DIR, DEVICE)
        model = fine_tune_model(model, tokenizer, TRAIN_FILE, OUTPUT_DIR, EPOCHS, BATCH_SIZE)
        logging.info("✅ Fine tuning completato. Per eseguire inference sul modello fine tuned, imposta MODEL_PATH con il valore OUTPUT_DIR.")
    else:
        logging.error("MODE non valido. Imposta MODE a 'inference' o 'train'.")

if __name__ == "__main__":
    main()
