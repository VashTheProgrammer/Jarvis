# Jarvis

## Introduzione
Jarvis è un assistente AI basato su tecnologie di **Machine Learning** e **Natural Language Processing (NLP)**. L'obiettivo del progetto è offrire un'interfaccia intelligente per rispondere alle domande degli utenti e fornire informazioni basate su un modello addestrato. 

Il progetto utilizza tecniche di **Retrieval-Augmented Generation (RAG)**, che combinano la generazione di testo con il recupero di informazioni da una base di conoscenza. Ciò migliora la pertinenza e la precisione delle risposte.

## Installazione di Miniconda
Miniconda è una versione leggera di Anaconda che consente di gestire ambienti Python isolati. Questo aiuta a evitare conflitti tra le librerie installate e fornisce un ambiente riproducibile.

### Perché usare Miniconda?
- Permette di creare **ambienti virtuali** per isolare le dipendenze.
- Riduce i problemi di compatibilità tra versioni di Python e librerie.
- Integra strumenti avanzati per la gestione dei pacchetti.

### Installazione:
1. **Scarica Miniconda** dal sito ufficiale: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. **Installa Miniconda**:
   - **Windows**: Esegui l'installer `.exe` e segui le istruzioni.
   - **macOS/Linux**: Apri il terminale ed esegui:
     ```sh
     bash Miniconda3-latest-MacOSX-x86_64.sh  # Per macOS Intel
     bash Miniconda3-latest-MacOSX-arm64.sh   # Per macOS Apple Silicon
     bash Miniconda3-latest-Linux-x86_64.sh   # Per Linux
     ```
3. **Verifica l'installazione** con:
   ```sh
   conda --version
   ```

### Creazione di un ambiente virtuale:
```sh
conda create -n jarvis_env python=3.10
conda activate jarvis_env
```

## Integrazione con Visual Studio Code
Per utilizzare Miniconda su **Visual Studio Code (VSC)**, segui questi passaggi:

1. **Installa l'estensione Python** da Microsoft (se non l'hai già fatto).
2. **Apri il terminale in VSC** e attiva l'ambiente virtuale:
   ```sh
   conda activate jarvis_env
   ```
3. **Seleziona il kernel corretto**:
   - Premi **Ctrl+Shift+P**.
   - Digita "Python: Select Interpreter".
   - Seleziona l'interprete relativo all'ambiente `jarvis_env`.

Questa configurazione ti permetterà di eseguire e debuggare gli script in un ambiente isolato senza interferenze con altre installazioni Python.

## Panoramica degli script
### `fine_tuning.py`
Questo script esegue il **fine-tuning** del modello AI utilizzando il dataset contenuto in `training_data.jsonl`. 

- **Scopo**: Adattare il modello alle esigenze specifiche del progetto.
- **Funzionamento**:
  1. Carica i dati da `training_data.jsonl`.
  2. Esegue il training incrementale sul modello esistente.
  3. Salva il modello aggiornato.
- **Esecuzione**:
  ```sh
  python fine_tuning.py
  ```

### `training_data.jsonl`
Questo file JSON contiene dati di addestramento. Ogni riga rappresenta un esempio di input-output per migliorare la capacità predittiva del modello.

Formato di esempio:
```json
{"input": "Come stai?", "output": "Sto bene, grazie!"}
```

### `run_model.py`
Esegue il modello AI per interagire con gli utenti.
- **Scopo**: Permette all'utente di inviare input e ricevere risposte generate dal modello.
- **Esecuzione**:
  ```sh
  python run_model.py
  ```

### `rag_model.py`
Implementa la metodologia **Retrieval-Augmented Generation (RAG)**.
- **Scopo**: Migliorare la precisione del modello combinando generazione di testo e recupero di dati.
- **Esecuzione**:
  ```sh
  python rag_model.py
  ```

### `performance_test.py`
Valuta le prestazioni del modello, misurando:
- **Tempo di risposta**
- **Precisione delle predizioni**
- **Consumo di memoria**

Esecuzione:
```sh
python performance_test.py
```

## Installazione delle dipendenze
Le librerie richieste possono essere installate con:
```sh
pip install -r requirements.txt
```
Se `requirements.txt` non è disponibile, puoi installare manualmente le librerie principali:
```sh
pip install torch transformers datasets numpy jsonlines
```

## Differenze tra macOS Intel, Apple Silicon e GPU
### Supporto per PyTorch
- **macOS Intel**: Supporto CPU standard, GPU non ottimizzata.
- **macOS Apple Silicon (M1/M2)**: Supporto via Metal (limitato rispetto a CUDA).
- **GPU NVIDIA (Linux/Windows)**: Supporto completo per CUDA.

Verifica il supporto GPU:
```sh
python -c "import torch; print(torch.cuda.is_available())"
```

## Esecuzione del progetto
Per avviare l'assistente Jarvis:
```sh
python run_model.py
```
Per testare le prestazioni:
```sh
python performance_test.py
```

## Esempio di `requirements.txt`
```txt
torch
transformers
datasets
numpy
jsonlines
scipy
tqdm
```

## Conclusione
Jarvis è un assistente AI avanzato che utilizza tecnologie di NLP e Machine Learning. Segui questa guida per configurarlo correttamente e adattarlo alle tue esigenze.


 
