import os

os.environ["HF_HOME"] = "/Users/adrianocostanzo/IA/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/Users/adrianocostanzo/IA/Datasets"

import time
import torch
import gc
import logging
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# ============================================
# Configurazioni (modifica direttamente qui)
# ============================================

# 📌 Scegli il modello RAG da eseguire.
# Puoi usare "facebook/rag-token-base" oppure "facebook/rag-sequence-base".
MODEL_NAME = "facebook/rag-sequence-base"

# 📌 Imposta la directory in cui salvare i modelli (cache_dir)
MODEL_DIR = os.path.expanduser("/Users/adrianocostanzo/IA/Models")  # Modifica per la tua directory preferita

# 📌 Imposta il percorso del file contenente il corpus (passages) in formato JSON.
# Ad esempio, il file potrebbe essere una lista di documenti strutturati come:
# [{"title": "Assistant Identity", "text": "You are AIDan, a friendly assistant...", "id": "1"}, ...]
PASSAGES_PATH = os.path.expanduser("/Users/adrianocostanzo/IA/my_corpus.json")

TOKEN_COUNT = 100  # Numero di token da generare per il test

# 📌 Prompt di input tradotto in inglese e leggermente modificato per stimolare una risposta
PROMPT = "Hello, who are you? Who am I speaking with? Give me a direct and concise answer. Answer:"

# ============================================

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Forza l'uso della CPU (ignora CUDA e MPS)
DEVICE = "cpu"
logging.info(f"Dispositivo forzato: {DEVICE}")

def clear_memory():
    """Libera la memoria: esegue gc.collect() e svuota la cache (se presente)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_rag_model(model_name, passages_path, device, cache_dir):
    """
    Carica il modello RAG e il retriever utilizzando il corpus specificato.
    Viene utilizzato il parametro cache_dir per salvare i modelli e i tokenizer nella directory scelta.
    Passiamo anche trust_remote_code=True per permettere l'esecuzione di codice custom dal repository.
    """
    logging.info(f"📥 [1/3] Downloading & Loading RAG model: {model_name}...")
    clear_memory()
    
    # Carica il tokenizer RAG specificando cache_dir e trust_remote_code=True
    tokenizer = RagTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    
    # Carica il retriever RAG specificando cache_dir, il percorso dei passaggi e trust_remote_code=True
    retriever = RagRetriever.from_pretrained(
        model_name,
        index_name="exact",       # Metodo di indicizzazione predefinito
        passages_path=passages_path,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    # Carica il modello RAG associato al retriever, specificando cache_dir e trust_remote_code=True
    model = RagSequenceForGeneration.from_pretrained(
        model_name,
        retriever=retriever,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    model.eval()
    
    logging.info("✅ RAG model loaded!")
    return model, tokenizer

def generate_text_rag(model, tokenizer, prompt, max_new_tokens):
    """
    Genera testo utilizzando il modello RAG.
    
    Prepara gli input, esegue la generazione con alcuni parametri (do_sample, temperature, ecc.)
    e poi libera le variabili temporanee per ottimizzare l'allocazione della memoria.
    """
    logging.info(f"🚀 [2/3] Running RAG inference for {max_new_tokens} tokens...")
    
    # Prepara gli input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    end_time = time.perf_counter()
    
    response_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    inference_time = end_time - start_time
    tokens_per_second = max_new_tokens / inference_time if inference_time > 0 else 0
    logging.info(f"✅ Inference complete! Time taken: {inference_time:.2f} sec | Tokens/sec: {tokens_per_second:.2f}")
    
    # Ottimizzazione della memoria: elimina variabili temporanee e richiama la garbage collection
    del outputs, inputs
    clear_memory()
    
    return response_text, inference_time, tokens_per_second

def main():
    logging.info("🖥️ Starting RAG Model Execution Benchmark...")
    clear_memory()
    
    # Carica il modello RAG e il retriever, utilizzando il cache_dir specificato
    model, tokenizer = load_rag_model(MODEL_NAME, PASSAGES_PATH, DEVICE, MODEL_DIR)
    
    # Genera la risposta
    response, inference_time, tokens_per_second = generate_text_rag(model, tokenizer, PROMPT, TOKEN_COUNT)
    
    logging.info("📌 [3/3] RAG Model Execution Results:")
    print("----------------------------------------------------------")
    print("📝 Prompt:")
    print(PROMPT)
    print("\n🤖 Response:")
    print(response)
    print("----------------------------------------------------------")
    
    logging.info("✅ RAG Model Execution Complete!")

if __name__ == "__main__":
    main()
