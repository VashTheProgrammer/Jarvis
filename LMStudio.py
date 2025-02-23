import requests
import json

# URL del server API di LM Studio
LM_STUDIO_API_URL = "http://localhost:1234"

def get_active_model():
    """Interroga LM Studio per ottenere il modello attivo e stampa la risposta per debug"""
    try:
        response = requests.get(f"{LM_STUDIO_API_URL}/v1/models")
        response.raise_for_status()
        models_data = response.json()
        
        # 🔍 Debug: stampiamo il JSON per vedere cosa restituisce LM Studio
        print("\n🔍 DEBUG - Risposta API /v1/models:\n", json.dumps(models_data, indent=4))

        models = models_data.get("data", [])

        if not models:
            print("⚠️ Nessun modello caricato in LM Studio. Verifica che il modello sia attivo.")
            return None
        
        active_model = models[0].get("id", None)
        print(f"✅ Modello attivo rilevato: {active_model}")
        return active_model
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore nel recupero del modello: {e}")
        return None

def generate_text(prompt, max_tokens=300):
    """Esegue l'inferenza usando il modello attivo di LM Studio"""
    model_name = get_active_model()
    if not model_name:
        return "❌ Errore: Nessun modello attivo trovato."

    headers = {"Content-Type": "application/json"}

    # 🔥 Simuliamo un contesto simile a LM Studio
    system_message = (
        "Sei un assistente AI utile e amichevole di nome JARVIS. Rispondi alle domande in modo naturale e cordiale.\n\n"
    )

    # Formattiamo il prompt per essere più vicino a LM Studio
    formatted_prompt = f"{system_message}{prompt.strip()}\n\n"

    data = {
        "model": model_name,
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_last_n": 64,
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "n_ctx": 4096,
        "n_batch": 512,
        "n_keep": 37,
        "n_predict": -1,
        "dry_multiplier": 0.000,
        "dry_base": 1.750,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": -1,
        "xtc_probability": 0.000,
        "xtc_threshold": 0.100,
        "typical_p": 1.000,
        "top_n_sigma": -1.000,
        "mirostat": 0,
        "mirostat_lr": 0.100,
        "mirostat_ent": 5.000
    }

    # 🔍 Stampa il JSON della richiesta per debug
    print("\n🔍 DEBUG - JSON della richiesta API:\n", json.dumps(data, indent=4))

    try:
        response = requests.post(f"{LM_STUDIO_API_URL}/v1/completions", headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "").strip()
    except requests.exceptions.RequestException as e:
        return f"❌ Errore nella richiesta: {e}"

if __name__ == "__main__":
    prompt = input("Inserisci il prompt: ")
    risposta = generate_text(prompt)
    print("\n📢 Risposta del modello phi-4:\n", risposta)
