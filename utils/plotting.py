# utils/plotting.py

import pandas as pd
import numpy as np
import glob
import json
import os
from typing import Dict, List, Optional

# Optional Dependency Handling
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    pass



def _check_deps():
    """Private helper to raise a clear error if plotting libs are missing."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Plotting requires 'matplotlib' and 'seaborn'.\n"
            "Please install them by running: pip install \"evolvepy[analysis]\""
        )


def _load_and_process_tuning_data(
    results_dir: str, 
    baseline_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Private helper to load all raw tuning JSONs and
    process them into a single, long-form DataFrame.
    """
    _check_deps()
    
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json result files found in {results_dir}")

    all_run_data = []
    
    # 1. Loop through each strategy's result file
    for file_path in json_files:
        strategy_name = os.path.basename(file_path).replace(".json", "")
        
        with open(file_path, 'r') as f:
            strategy_data = json.load(f)
            
        # 2. Loop through each problem in that file
        for problem_name, run_histories in strategy_data.items():
            
            # 3. Loop through each of the N statistical runs
            for run_idx, history in enumerate(run_histories):
                
                # Get the baseline fitness for this problem
                baseline_fitness = baseline_dict.get(problem_name)
                if baseline_fitness is None:
                    continue # Skip problems that failed baseline (e.g., 'r' files)
                    
                # 4. Process the history time-series
                try:
                    generations = history['generation']
                    best_fitness_ts = history['best_fitness']
                    
                    # Normalize the *entire* fitness time-series
                    # (ea_fit - base_fit) / abs(base_fit)
                    improvement_ts = (
                        (np.array(best_fitness_ts) - baseline_fitness) / 
                        abs(baseline_fitness) * 100
                    )
                    
                    # Store one row for each generation
                    for gen, imp in zip(generations, improvement_ts):
                        all_run_data.append({
                            "strategy": strategy_name,
                            "problem": problem_name,
                            "run_id": run_idx,
                            "generation": gen,
                            "improvement_pct": imp
                        })
                except Exception as e:
                    print(f"Warning: Skipping malformed history for {strategy_name} on {problem_name}: {e}")

    if not all_run_data:
        raise ValueError("No valid, processed data could be generated from the results.")
        
    return pd.DataFrame(all_run_data)



def plot_strategy_comparison(
    data: pd.DataFrame, 
    x: str, 
    y: str, 
    title: str = "Strategy Performance Comparison",
    save_path: Optional[str] = None, 
    show: bool = True
):
    """
    Generates a Box Plot to compare the final performance
    distribution of multiple strategies.

    Args:
        data (pd.DataFrame): A long-form DataFrame containing the data.
        x (str): The column name for the x-axis (e.g., 'improvement_pct').
        y (str): The column name for the y-axis (e.g., 'strategy_name').
        title (str, optional): The title for the plot.
        save_path (str, optional): File path to save the plot.
        show (bool): Whether to display the plot.
    """
    _check_deps()
        
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Use a boxplot to show the full distribution of results
    sns.boxplot(data=data, x=x, y=y, orient="h")
    
    plt.title(title, fontsize=16)
    plt.xlabel(x.replace("_", " ").title(), fontsize=12)
    plt.ylabel(y.replace("_", " ").title(), fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_tuning_convergence(
    results_dir: str,
    baseline_file_path: str,
    top_k: Optional[int] = 5,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plots the mean convergence curves for the 'top_k' best strategies.

    This function loads all raw tuning data, calculates the *final*
    mean performance for every strategy, identifies the top 'k'
    performers, and then plots *only* their convergence curves
    (mean % improvement vs. generation).

    Args:
        results_dir (str): Path to the 'experiments/tuning_results' folder.
        baseline_file_path (str): Path to the 'data/baseline_hc.json' file.
        top_k (int, optional): The number of top strategies to plot.
            If None, all strategies will be plotted. Defaults to 5.
        save_path (str, optional): The file path to save the plot image.
        show (bool): Whether to display the plot interactively.
    """
    _check_deps()
    print("Loading and processing all tuning data...")
    
    # Load Data
    try:
        with open(baseline_file_path, 'r') as f:
            baseline_dict = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Baseline file not found at {baseline_file_path}")
        return

    # Process all raw files into a single, normalized DataFrame
    df = _load_and_process_tuning_data(results_dir, baseline_dict)
    
    # Rank and Filter for top_k 
    if top_k is not None:
        print(f"Finding the top {top_k} performing strategies...")
        
        # Find the *final* performance (last generation) for each run
        last_gen = df['generation'].max()
        df_final_performance = df[df['generation'] == last_gen]
        
        # Calculate the *mean* final performance for each strategy
        # (averaging across all problems and all statistical runs)
        strategy_ranking = df_final_performance.groupby('strategy')['improvement_pct'].mean()
        
        # Sort and get the top 'k' names
        strategy_ranking = strategy_ranking.sort_values(ascending=False)
        top_k_strategies = strategy_ranking.head(top_k).index
        
        # Filter the main DataFrame to *only* include these strategies
        df_to_plot = df[df['strategy'].isin(top_k_strategies)]
        
        print("Top strategies found:", list(top_k_strategies))
    else:
        print("Plotting all strategies (top_k=None).")
        df_to_plot = df
        top_k_strategies = df['strategy'].unique() # Plot in default order

    # Plotting
    print("Generating convergence plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    # 'seaborn.lineplot' automatically aggregates (means) and
    # creates a 95% confidence interval band (the shaded area).
    ax = sns.lineplot(
        data=df_to_plot,
        x='generation',
        y='improvement_pct',
        hue='strategy',
        hue_order=top_k_strategies, # Order the legend by performance
        errorbar=('ci', 95) # Show 95% confidence interval
    )

    ax.set_title(f"Mean Convergence of Top {top_k or 'All'} Strategies", fontsize=16)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Mean % Improvement over HC Baseline", fontsize=12)
    ax.legend(title="Strategy", loc='center left', bbox_to_anchor=(1, 0.5))
    ax.axhline(0, color='red', linestyle='--', label='HC Baseline (0%)')
    
    if save_path:
        # Use bbox_inches='tight' to ensure the legend is not cut off
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_improvement_by_size(
    df: pd.DataFrame, 
    save_path: Optional[str] = None, 
    show: bool = True
):
    """
    Generates a Box Plot of the % improvement vs. problem size.

    This plot is ideal for visualizing *how* the EA's performance
    (relative to the baseline) changes as problems get harder.
    
    Args:
        df (pd.DataFrame): The processed DataFrame containing
                           'problem_size' and 'mean_improvement_pct' columns.
        save_path (str, optional): The file path to save the plot image.
        show (bool): Whether to display the plot interactively.
    """
    _check_deps()

    print("Generating Plot 1: % Improvement vs. Problem Size...")
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create a box plot to show the distribution of results
    sns.boxplot(
        data=df,
        x='problem_size',
        y='mean_improvement_pct',
        order=sorted(df['problem_size'].unique()), # Ensure x-axis is sorted
        ax=ax
    )

    # Add a horizontal line at 0% for the baseline reference
    ax.axhline(0, color='red', linestyle='--', label='HC Baseline (0%)')

    ax.set_title("EA Performance vs. Problem Size", fontsize=16)
    ax.set_xlabel("Problem Size (Number of Cities)", fontsize=12)
    ax.set_ylabel("% Improvement over HC Baseline", fontsize=12)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)



def plot_absolute_fitness_comparison(
    df: pd.DataFrame, 
    save_path: Optional[str] = None, 
    show: bool = True
):
    """
    Generates a grouped bar chart comparing the EA's mean fitness
    against the HC baseline fitness for each problem size.

    Args:
        df (pd.DataFrame): The processed DataFrame containing
                           'problem_size', 'mean_fitness', and
                           'baseline_fitness_hc' columns.
        save_path (str, optional): The file path to save the plot image.
        show (bool): Whether to display the plot interactively.
    """
    _check_deps()
    
    print("Generating Plot 2: Absolute Fitness (EA vs. HC)...")

    # Pre-processing
    # We "melt" the DataFrame to make it easy for seaborn to plot
    # 'mean_fitness' and 'baseline_fitness_hc' as a grouped variable.
    df_melted = df.reset_index().melt(
        id_vars=['index', 'problem_size'],
        value_vars=['mean_fitness', 'baseline_fitness_hc'],
        var_name='Algorithm',
        value_name='Fitness'
    )
    
    # Rename for a clearer legend
    df_melted['Algorithm'] = df_melted['Algorithm'].map({
        'mean_fitness': 'EA (Mean of N Runs)',
        'baseline_fitness_hc': 'HC Baseline (Best of N Runs)'
    })

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create a grouped bar chart
    sns.barplot(
        data=df_melted,
        x='problem_size',
        y='Fitness',
        hue='Algorithm',
        order=sorted(df['problem_size'].unique()),
        ax=ax
    )

    ax.set_title("Absolute Fitness: EA vs. Hill Climber Baseline", fontsize=16)
    ax.set_xlabel("Problem Size (Number of Cities)", fontsize=12)
    ax.set_ylabel("Fitness (Negative Distance)", fontsize=12)
    ax.legend(title="Algorithm")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)