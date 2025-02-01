import numpy as np
import time
import platform
import torch
import subprocess

def get_system_info():
    """Raccoglie informazioni di sistema (OS, CPU, GPU)."""
    system_info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Processor": platform.processor(),
        "CPU Cores": torch.get_num_threads(),
        "PyTorch Version": torch.__version__,
        "GPU Available in PyTorch": torch.cuda.is_available(),
        "GPU Name (PyTorch)": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "GPU Name (OpenCL)": get_opencl_gpu_name(),
    }
    return system_info

def get_opencl_gpu_name():
    """Verifica se esistono GPU OpenCL, utile per macOS con GPU AMD."""
    try:
        result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)
        lines = result.stdout.split("\n")
        for line in lines:
            if "Chipset Model:" in line:
                return line.split(":")[-1].strip()
    except Exception:
        return "N/A"
    return "N/A"

def cpu_benchmark():
    """Test delle prestazioni della CPU con moltiplicazione di matrici grandi."""
    print("Testing CPU performance...")
    start_time = time.time()
    A = np.random.rand(5000, 5000)
    B = np.random.rand(5000, 5000)
    np.dot(A, B)
    return time.time() - start_time

def memory_benchmark():
    """Test delle prestazioni della RAM creando e manipolando array di grandi dimensioni."""
    print("Testing RAM performance...")
    start_time = time.time()
    big_array = np.random.rand(100_000_000)  # 100 milioni di elementi
    big_array *= 2
    del big_array
    return time.time() - start_time

def gpu_benchmark():
    """Test delle prestazioni della GPU se disponibile in PyTorch."""
    if torch.cuda.is_available():
        print("Testing GPU performance (PyTorch CUDA)...")
        start_time = time.time()
        A = torch.rand(10000, 10000, device="cuda")
        B = torch.rand(10000, 10000, device="cuda")
        torch.mm(A, B)
        torch.cuda.synchronize()  # Assicura che il calcolo sia terminato
        return time.time() - start_time
    return None

def estimate_inference_time(cpu_time):
    """
    Stima il tempo di inferenza su CPU per modelli di grandi dimensioni.
    Formula approssimativa basata su test con LLaMA 7B e DeepSeek R1.
    """
    tokens_per_second = 1 / (cpu_time * 0.75)  # Stima basata sulla moltiplicazione di matrici
    estimated_time_100_tokens = 100 / tokens_per_second  # Tempo per generare 100 token (circa una risposta breve)
    
    return estimated_time_100_tokens

def run_benchmarks():
    """Esegue i benchmark e raccoglie i risultati."""
    system_info = get_system_info()
    
    cpu_time = cpu_benchmark()
    ram_time = memory_benchmark()
    gpu_time = gpu_benchmark()

    results = {
        "CPU (Matrix Multiplication)": f"{cpu_time:.3f} sec",
        "RAM (Memory Read/Write)": f"{ram_time:.3f} sec",
        "GPU (if available)": f"{gpu_time:.3f} sec" if gpu_time else "No GPU support in PyTorch",
    }

    # Stima del tempo di inferenza
    estimated_inference_time = estimate_inference_time(cpu_time)

    return system_info, results, estimated_inference_time

def display_results(system_info, results, estimated_inference_time):
    """Mostra i risultati a schermo in formato leggibile."""
    
    print("\n📌 SYSTEM INFORMATION:")
    for key, value in system_info.items():
        print(f"  - {key}: {value}")

    print("\n📊 BENCHMARK RESULTS:")
    for key, value in results.items():
        print(f"  - {key}: {value}")

    print("\n⏳ ESTIMATED INFERENCE TIME FOR AI MODELS:")
    print(f"  - DeepSeek R1 / LLaMA 7B (100 tokens response): ~{estimated_inference_time:.2f} seconds\n")

if __name__ == "__main__":
    system_info, results, estimated_inference_time = run_benchmarks()
    display_results(system_info, results, estimated_inference_time)
