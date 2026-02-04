import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_convergence(stats_dir, output_path):
    stats_dir = Path(stats_dir)
    if not stats_dir.exists():
        print(f"Directory not found: {stats_dir}")
        return

    # Metrics to track
    metrics = {
        'prob_diff': {'data': {}, 'label': 'Prob Diff (L2)', 'color': 'royalblue', 'marker': 'o'},
        'cos_sim': {'data': {}, 'label': 'Cosine Similarity', 'color': 'teal', 'marker': 's'},
        'entropy': {'data': {}, 'label': 'Entropy', 'color': 'crimson', 'marker': '^'}
    }

    # Load all json files
    for json_file in stats_dir.glob("*_stats.json"):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
                
            for ex_idx, steps in data.items():
                for step_info in steps:
                    step = step_info['step']
                    
                    for m_key in metrics:
                        if m_key in step_info:
                            val = step_info[m_key]
                            # Skip invalid markers
                            if val < -0.5 and m_key == 'prob_diff': continue 
                            
                            if step not in metrics[m_key]['data']:
                                metrics[m_key]['data'][step] = []
                            metrics[m_key]['data'][step].append(val)

    # Filter out metrics with no data
    metrics = {k: v for k, v in metrics.items() if v['data']}
    
    if not metrics:
        print("No valid convergence data found.")
        return

    # Plot
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (m_key, m_info) in zip(axes, metrics.items()):
        data_dict = m_info['data']
        sorted_steps = sorted(data_dict.keys())
        
        means = np.array([np.mean(data_dict[s]) for s in sorted_steps])
        stds = np.array([np.std(data_dict[s]) for s in sorted_steps])
        
        ax.plot(sorted_steps, means, marker=m_info['marker'], linestyle='-', linewidth=2, 
                color=m_info['color'], label=f'Mean {m_info["label"]} ($\mu$)')
        ax.plot(sorted_steps, stds, marker='x', linestyle='--', linewidth=1, 
                color='orange', alpha=0.7, label=f'Std Dev ($\sigma$)')

        ax.set_ylabel('Value', fontsize=18)
        ax.set_title(f'Metric: {m_info["label"]}', fontsize=20)
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(
            loc='best',
            fontsize=16,
            frameon=True,
            framealpha=0.95,
            facecolor='white',
            edgecolor='#CCCCCC',
            borderpad=0.6,
            labelspacing=0.5,
            handlelength=2.2,
        )

        # Scaling
        if m_key in ['prob_diff', 'cos_sim']:
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
        else:
            # Entropy or others: dynamic
            pass

    axes[-1].set_xlabel('Loop Step', fontsize=18)
    plt.suptitle('Model Convergence Analysis', fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=str, required=True, help="Path to convergence_stats folder")
    parser.add_argument("--output", type=str, default="convergence_plot.png", help="Output plot filename")
    args = parser.parse_args()
    
    plot_convergence(args.stats_dir, args.output)
