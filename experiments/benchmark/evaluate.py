from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import optuna
import optunahub
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


def get_tqdm():
    """Return the appropriate tqdm instance based on the environment."""
    try:
        import IPython
        if IPython.get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return tqdm_notebook
    except (NameError, ImportError):
        pass
    return tqdm


class Evaluator:
    def __init__(self) -> None:
        self.results = {}
        self.tqdm = get_tqdm()
        
        self.default_configs = {
            "bbob": [
                {"function_id": i, "dimension": d} 
                for i in range(1, 25)
                for d in [2, 5, 10]
            ],
            "bbob_noisy": [
                {"function_id": i, "dimension": d}
                for i in range(101, 131)
                for d in [2, 5, 10]
            ],
            "zdt": [
                {"function_id": i}
                for i in range(1, 7)
            ]
        }

    def run(
        self,
        samplers: List[optuna.samplers.BaseSampler],
        benchmarks: Optional[List[str]] = None,
        n_trials: int = 100,
        verbose: bool = False,
    ) -> None:
        """Run benchmark evaluation across different samplers.

        Args:
            samplers: List of Optuna samplers to evaluate
            benchmarks: List of benchmark names to run. If None, runs all default benchmarks
            n_trials: Number of trials per optimization run
            verbose: Whether to enable detailed logging
        """
        if benchmarks is None:
            benchmarks = ["bbob", "bbob_noisy", "zdt"]

        optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)

        table_data = []
        headers = ["Benchmark", "Config", "Sampler", "Best Value"]
        
        total_runs = sum(len(self.default_configs[b]) * len(samplers) for b in benchmarks)
        progress_bar = self.tqdm(total=total_runs, desc="Running benchmarks")

        for bench_name in benchmarks:            
            bench_module = optunahub.load_module(f"benchmarks/{bench_name}")
            configs = self.default_configs[bench_name]
            
            self.results[bench_name] = {
                "configs": configs,
                "sampler_results": {str(s.__class__.__name__): [] for s in samplers}
            }

            for config in configs:
                problem = bench_module.Problem(**config)

                for sampler in samplers:
                    sampler_name = str(sampler.__class__.__name__)
                    
                    study = optuna.create_study(
                        sampler=sampler,
                        directions=problem.directions
                    )
                    study.optimize(problem, n_trials=n_trials, show_progress_bar=False)

                    if len(problem.directions) > 1:
                        pareto_front = [t.values for t in study.best_trials]
                        best_value = np.mean(pareto_front) if pareto_front else float('inf')
                    else:
                        best_value = study.best_value
                    
                    self.results[bench_name]["sampler_results"][sampler_name].append({
                        "config": config,
                        "value": best_value
                    })
                    
                    config_str = ", ".join(f"{k}={v}" for k, v in config.items())
                    table_data.append([
                        bench_name,
                        config_str,
                        sampler_name,
                        f"{best_value:.4e}"
                    ])
                    
                    progress_bar.update(1)

        progress_bar.close()
        print("\n## Benchmark Results\n")
        print(tabulate(table_data, headers=headers, tablefmt="pipe"))

    def plot(
        self,
        path: Optional[str] = None,
        figsize: tuple[int, int] = (12, 8)
    ) -> None:
        """Create comparison plots across all samplers for each benchmark.

        Args:
            path: Optional path to save the generated plots
            figsize: Figure size for the plots
        """
        for bench_name, bench_results in self.results.items():
            configs = bench_results["configs"]
            sampler_results = bench_results["sampler_results"]
            
            plt.figure(figsize=figsize)

            n_configs = len(configs)
            x = np.arange(n_configs)
            width = 0.8 / len(sampler_results)
            
            for i, (sampler_name, results) in enumerate(sampler_results.items()):
                values = [r["value"] for r in results]
                plt.bar(x + i * width, values, width, label=sampler_name)
            
            plt.xlabel("Problem Configuration")
            plt.ylabel("Objective Value")
            plt.title(f"{bench_name} Benchmark Results")
            plt.legend()
            
            labels = [f"{i+1}" for i in range(n_configs)]
            plt.xticks(x + width * (len(sampler_results) - 1) / 2, labels)
            
            if path:
                save_dir = Path(path)
                save_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_dir / f"{bench_name}_comparison.png")
            
            plt.close()
