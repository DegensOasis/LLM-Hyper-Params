import itertools
from typing import Dict, List, Tuple, Callable
import numpy as np

class ParameterExplorer:
    def __init__(self, param_ranges: Dict[str, Tuple[float, float, float]]):
        self.param_ranges = param_ranges
        
    def grid_search(self) -> List[Dict[str, float]]:
        param_values = [np.arange(start, end + step, step) for start, end, step in self.param_ranges.values()]
        combinations = list(itertools.product(*param_values))
        return [dict(zip(self.param_ranges.keys(), combo)) for combo in combinations]
    
    def random_search(self, n_samples: int) -> List[Dict[str, float]]:
        samples = []
        for _ in range(n_samples):
            sample = {
                param: np.random.uniform(start, end)
                for param, (start, end, _) in self.param_ranges.items()
            }
            samples.append(sample)
        return samples

    def explore(self, generate_function: Callable, prompt: str, method: str = 'grid', n_samples: int = 10) -> List[Tuple[Dict[str, float], Tuple[str, float]]]:
        if method == 'grid':
            param_combinations = self.grid_search()
        elif method == 'random':
            param_combinations = self.random_search(n_samples)
        else:
            raise ValueError("Invalid search method. Choose 'grid' or 'random'.")
        
        results = []
        for params in param_combinations:
            output, score = generate_function(prompt, **params)
            if output is not None:
                results.append((params, (output, score)))
        
        return results
