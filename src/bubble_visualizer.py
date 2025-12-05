"""
Visualizzazione risultati VERABENCH

Genera grafici bubble chart per analizzare performance per task:
- X: Costo per Esempio (USD)
- Y: Accuratezza (%) - metrica specifica per task
- Size: Latenza Media (secondi)
- Color: Modello

Task-specific accuracy metrics:
- final_answer: overall_quality
- judge: judgment_accuracy
- rag: retrieval_accuracy
- routing: routing_accuracy
- tool_calling: tool_selection_accuracy
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary dataframe from results."""
    summary_data = []

    for result in results:
        task = result['task']
        metrics = result['metrics']
        config = result.get('config', {})

        # Map accuracy based on task type
        accuracy_map = {
            'final_answer': 'overall_quality',
            'judge': 'judgment_accuracy',
            'rag': 'retrieval_accuracy',
            'routing': 'routing_accuracy',
            'tool_calling': 'tool_selection_accuracy'
        }

        # Get the appropriate accuracy metric for this task
        accuracy_field = accuracy_map.get(task, 'accuracy')
        accuracy = metrics.get(accuracy_field, 0.0)

        # If accuracy is already a percentage (0-100), keep it; if it's a ratio (0-1), convert
        if accuracy <= 1.0:
            accuracy *= 100

        # Get total_examples to calculate avg_latency
        total_examples = metrics.get('total_examples', 1)
        total_latency = metrics.get('total_latency', 0.0)
        avg_latency = total_latency / max(total_examples, 1)

        # Get total cost
        total_cost = metrics.get('total_cost', 0.0)

        summary_data.append({
            'task': task,
            'model': config.get('model_name', result.get('model', 'unknown')),
            'variant': result.get('variant', 'default').replace('_variant', '').upper(),
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'total_cost': total_cost,
            'total_examples': total_examples
        })

    return pd.DataFrame(summary_data)


def calculate_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cost per example from total cost."""
    # Use the total_cost already calculated by the metrics
    df['cost_usd'] = df['total_cost']

    # Calculate average cost per example
    df['cost_per_example'] = df['total_cost'] / df['total_examples'].replace(0, 1)

    return df


def create_task_bubble_chart(df: pd.DataFrame, task: str, output_dir: Path):
    task_df = df[df['task'] == task].copy()

    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color map for models
    models = sorted(task_df['model'].unique())
    color_palette = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    model_colors = {model: color_palette[i % len(color_palette)] for i, model in enumerate(models)}

    # Marker map for variants (default to circle if no variant specified)
    variant_markers = {
        'JSON': 'o',      # Circle
        'XML': 's',       # Square
        'COT': '^',       # Triangle
        'DEFAULT': 'o'    # Circle for default
    }

    # Normalize bubble sizes (latency)
    min_latency = task_df['avg_latency'].min()
    max_latency = task_df['avg_latency'].max()
    size_scale = 3000  # Base size for bubbles

    # Plot each combination
    for _, row in task_df.iterrows():
        # Normalize latency to bubble size
        if max_latency > min_latency:
            norm_latency = (row['avg_latency'] - min_latency) / (max_latency - min_latency)
        else:
            norm_latency = 0.5
        bubble_size = size_scale * (0.3 + norm_latency * 0.7)

        marker = variant_markers.get(row['variant'], 'o')
        color = model_colors[row['model']]

        ax.scatter(
            row['cost_per_example'],
            row['accuracy'],
            s=bubble_size,
            c=[color],
            marker=marker,
            alpha=0.6,
            edgecolors='black',
            linewidth=2
        )

        # Add text label (simplified for clarity)
        label = f"{row['model']}"
        ax.annotate(
            label,
            (row['cost_per_example'], row['accuracy']),
            fontsize=8,
            ha='center',
            va='center',
            weight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none')
        )

    # Set labels and title
    ax.set_xlabel('Cost per Example (USD)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Task: {task.upper()} - Model Performance Analysis\n(Bubble Size = Average Latency)',
                 fontsize=15, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # Add quadrant lines at median
    if len(task_df) > 1:
        median_cost = task_df['cost_per_example'].median()
        median_accuracy = task_df['accuracy'].median()
        ax.axvline(median_cost, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.axhline(median_accuracy, color='red', linestyle='--', alpha=0.4, linewidth=1.5)

        # Annotate optimal quadrant
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(xlim[0] + (median_cost - xlim[0]) * 0.5, ylim[1] - (ylim[1] - ylim[0]) * 0.05,
                'OPTIMAL ZONE\nHigh Accuracy\nLow Cost',
                ha='center', va='top', fontsize=10, color='darkgreen', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.4, edgecolor='green', linewidth=2))
    
    # Create legends outside the plot
    # Legend 1: Models (colors)
    model_handles = []
    for model in models:
        handle = plt.scatter([], [], s=150, c=model_colors[model], marker='o',
                           edgecolors='black', linewidth=1.5, label=model, alpha=0.6)
        model_handles.append(handle)

    legend1 = ax.legend(handles=model_handles, title='Model',
                       bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=11,
                       framealpha=0.95, edgecolor='black', fancybox=True)
    ax.add_artist(legend1)

    # Legend 2: Latency (bubble sizes) - text only without markers
    from matplotlib.lines import Line2D
    size_handles = [
        Line2D([0], [0], marker='', color='w', label=f'Small  ({min_latency:.2f}s)',
               markerfacecolor='none', markersize=0),
        Line2D([0], [0], marker='', color='w', label=f'Medium ({(min_latency + max_latency) / 2:.2f}s)',
               markerfacecolor='none', markersize=0),
        Line2D([0], [0], marker='', color='w', label=f'Large  ({max_latency:.2f}s)',
               markerfacecolor='none', markersize=0)
    ]

    legend2 = ax.legend(handles=size_handles, title='Average Latency (Bubble Size)',
                       bbox_to_anchor=(1.02, 0.65), loc='upper left', fontsize=9, title_fontsize=10,
                       framealpha=0.95, edgecolor='black', fancybox=True)
    
    # Adjust layout to accommodate legends
    plt.tight_layout()
    
    # Save figure with extra space for legends
    output_path = output_dir / f"{task}_bubble_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Bubble chart saved: {output_path}")
    plt.close()


def print_task_summary(df: pd.DataFrame, task: str):
    """Print summary table for a specific task."""
    task_df = df[df['task'] == task].copy()

    print(f"\n{'='*80}")
    print(f"[TASK] {task.upper()} - RESULTS")
    print(f"{'='*80}\n")

    # Find best performers
    best_accuracy_idx = task_df['accuracy'].idxmax()
    best_accuracy = task_df.loc[best_accuracy_idx]

    fastest_idx = task_df['avg_latency'].idxmin()
    fastest = task_df.loc[fastest_idx]

    cheapest_idx = task_df['cost_per_example'].idxmin()
    cheapest = task_df.loc[cheapest_idx]

    print("[BEST] TOP PERFORMERS:")
    print(f"  - Accuracy: {best_accuracy['accuracy']:.1f}% - {best_accuracy['model']} ({best_accuracy['variant']})")
    print(f"  - Speed: {fastest['avg_latency']:.3f}s - {fastest['model']} ({fastest['variant']})")
    print(f"  - Cost per Example: ${cheapest['cost_per_example']:.6f} - {cheapest['model']} ({cheapest['variant']})")

    print(f"\n[DETAIL] ALL RESULTS:")
    print(f"{'Model':<20} {'Variant':<10} {'Accuracy %':<12} {'Latency (s)':<15} {'Cost/Example':<15}")
    print("-" * 80)

    for _, row in task_df.iterrows():
        print(f"{row['model']:<20} {row['variant']:<10} {row['accuracy']:>10.1f}% "
              f"{row['avg_latency']:>13.3f}s ${row['cost_per_example']:>13.6f}")


def visualize_results(results_data: List[Dict[str, Any]], task: str, output_dir: Path):
    if not results_data:
        print(f"No results to visualize")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataframe
    df = create_summary_dataframe(results_data)
    df = calculate_costs(df)

    # Filter for this task
    task_df = df[df['task'] == task]
    if task_df.empty:
        print(f"[X] No results found for task: {task}")
        return

    # Print summary
    print_task_summary(df, task)

    # Create bubble chart
    print(f"\n[*] Generating bubble chart...")
    create_task_bubble_chart(df, task, output_dir)

    print(f"\n{'='*80}")
    print("[OK] VISUALIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"[*] Chart saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualizza risultati Phase 1 Screening per una Task"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Nome della task da visualizzare"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path al file JSON dei risultati"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory di output per le visualizzazioni"
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"✗ File risultati non trovato: {results_path}")
        return
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "visualizations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"\n[*] Loading results from: {results_path}")
    results = load_results(results_path)
    df = create_summary_dataframe(results)
    df = calculate_costs(df)
    
    # Filter for this task
    task_df = df[df['task'] == args.task]
    if task_df.empty:
        print(f"[X] No results found for task: {args.task}")
        return
    
    # Print summary
    print_task_summary(df, args.task)
    
    # Create bubble chart
    print(f"\n[*] Generating bubble chart...")
    create_task_bubble_chart(df, args.task, output_dir)
    
    print(f"\n{'='*80}")
    print("[OK] ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"[*] Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
