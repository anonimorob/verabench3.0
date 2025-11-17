"""
Script per eseguire tutti i benchmark di VERABENCH in sequenza.

Esegue automaticamente:
1. Routing
2. Tool Calling
3. Judge
4. RAG
5. Final Answer

Ogni task viene eseguita con i modelli configurati nei rispettivi main_*.py.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Lista di tutti i benchmark da eseguire
BENCHMARKS = [
    {
        "name": "Routing",
        "script": "main_routing.py",
        "description": "Selezionare l'agente corretto per richieste utente",
    },
    {
        "name": "Tool Calling",
        "script": "main_tool_calling.py",
        "description": "Selezionare tool e parametri corretti",
    },
    {
        "name": "Judge",
        "script": "main_judge.py",
        "description": "Validare output prima di inviarli all'utente",
    },
    {
        "name": "RAG",
        "script": "main_rag.py",
        "description": "Recuperare dati da database interno con security check",
    },
    {
        "name": "Final Answer",
        "script": "main_final_answer.py",
        "description": "Generare risposte user-friendly per WhatsApp",
    },
]


def print_header():
    """Stampa header dello script."""
    print("=" * 70)
    print(" " * 20 + "VERABENCH - RUN ALL TASKS")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBenchmarks da eseguire: {len(BENCHMARKS)}")
    for i, benchmark in enumerate(BENCHMARKS, 1):
        print(f"  {i}. {benchmark['name']}: {benchmark['description']}")
    print("\n" + "=" * 70 + "\n")


def run_benchmark(benchmark: dict, index: int, total: int) -> dict:
    """
    Esegue un singolo benchmark.
    
    Args:
        benchmark: Dizionario con info benchmark
        index: Indice corrente (1-based)
        total: Totale benchmark
    
    Returns:
        Dizionario con risultati esecuzione
    """
    print(f"\n{'='*70}")
    print(f"[{index}/{total}] Running: {benchmark['name']}")
    print(f"Script: {benchmark['script']}")
    print(f"Description: {benchmark['description']}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    try:
        # Esegui il benchmark con uv run
        result = subprocess.run(
            ["uv", "run", "python", benchmark['script']],
            capture_output=False,  # Mostra output in tempo reale
            text=True,
            cwd=Path.cwd(),
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        success = result.returncode == 0
        
        print(f"\n{'='*70}")
        if success:
            print(f"✓ {benchmark['name']} completato con successo")
        else:
            print(f"✗ {benchmark['name']} fallito con exit code {result.returncode}")
        print(f"Durata: {duration:.1f}s")
        print(f"{'='*70}\n")
        
        return {
            "name": benchmark['name'],
            "script": benchmark['script'],
            "success": success,
            "duration": duration,
            "exit_code": result.returncode,
        }
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"✗ ERRORE durante esecuzione {benchmark['name']}")
        print(f"Errore: {str(e)}")
        print(f"{'='*70}\n")
        
        return {
            "name": benchmark['name'],
            "script": benchmark['script'],
            "success": False,
            "duration": duration,
            "error": str(e),
        }


def print_summary(results: list, total_duration: float):
    """
    Stampa summary finale di tutti i benchmark.
    
    Args:
        results: Lista di risultati
        total_duration: Durata totale in secondi
    """
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARY")
    print("=" * 70 + "\n")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total benchmarks: {len(results)}")
    print(f"Successful: {successful} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)\n")
    
    print("Results per benchmark:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{i}. {result['name']:<20} {status:<15} {result['duration']:.1f}s")
    
    print("-" * 70)
    
    if failed > 0:
        print("\nFailed benchmarks:")
        for result in results:
            if not result['success']:
                error_msg = result.get('error', f"Exit code: {result.get('exit_code', 'unknown')}")
                print(f"  - {result['name']}: {error_msg}")
    
    print("\n" + "=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def main():
    """Funzione principale."""
    print_header()
    
    # Conferma dall'utente
    try:
        response = input("Vuoi procedere con l'esecuzione di tutti i benchmark? (y/n): ")
        if response.lower() not in ['y', 'yes', 's', 'si', 'sì']:
            print("\nEsecuzione annullata dall'utente.")
            return
    except KeyboardInterrupt:
        print("\n\nEsecuzione annullata dall'utente.")
        return
    
    print("\nAvvio esecuzione benchmark...\n")
    
    start_time = datetime.now()
    results = []
    
    # Esegui tutti i benchmark
    for i, benchmark in enumerate(BENCHMARKS, 1):
        result = run_benchmark(benchmark, i, len(BENCHMARKS))
        results.append(result)
        
        # Pausa tra un benchmark e l'altro (opzionale)
        if i < len(BENCHMARKS):
            print(f"\nPausa di 2 secondi prima del prossimo benchmark...\n")
            import time
            time.sleep(2)
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    # Stampa summary
    print_summary(results, total_duration)
    
    # Exit code: 0 se tutti successo, 1 se almeno uno fallito
    exit_code = 0 if all(r['success'] for r in results) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
