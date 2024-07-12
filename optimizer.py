from typing import Tuple, Dict, List
from .explorer import ParameterExplorer
from .scorer import Scorer
from .models.meta_predictor import MetaPredictor, extract_prompt_features
import numpy as np
import random

class Optimizer:
    def __init__(self, explorer: ParameterExplorer, scorer: Scorer, llm_wrapper, meta_predictor: MetaPredictor, 
                 max_iterations: int = 5, max_recursion_depth: int = 3):
        self.explorer = explorer
        self.scorer = scorer
        self.llm_wrapper = llm_wrapper
        self.meta_predictor = meta_predictor
        self.max_iterations = max_iterations
        self.max_recursion_depth = max_recursion_depth

    def rephrase_prompt(self, prompt: str, **params) -> str:
        rephrase_instruction = f"Rephrase the following prompt to make it more clear and specific: '{prompt}'"
        return self.llm_wrapper.generate(rephrase_instruction, **params)

    def optimize(self, original_prompt: str, reference: str = None, method: str = 'grid', n_samples: int = 10) -> Tuple[Dict[str, float], str, str]:
        return self._optimize_recursive(original_prompt, reference, method, n_samples, recursion_depth=0)

    def _optimize_recursive(self, prompt: str, reference: str, method: str, n_samples: int, recursion_depth: int) -> Tuple[Dict[str, float], str, str]:
        if recursion_depth >= self.max_recursion_depth:
            return self._optimize_iterative(prompt, reference, method, n_samples)

        best_overall_params = None
        best_overall_output = None
        best_overall_score = float('-inf')
        best_overall_prompt = prompt

        for iteration in range(self.max_iterations):
            print(f"Recursion depth: {recursion_depth}, Iteration {iteration + 1}/{self.max_iterations}")
            
            if iteration > 0:
                best_overall_prompt = self.rephrase_prompt(best_overall_prompt, **best_overall_params)
                print(f"Rephrased prompt: {best_overall_prompt}")

            current_params, current_output, current_score = self._optimize_iterative(best_overall_prompt, reference, method, n_samples)

            if current_score > best_overall_score:
                best_overall_params = current_params
                best_overall_output = current_output
                best_overall_score = current_score
                best_overall_prompt = best_overall_prompt
            else:
                # If no improvement, try a recursive call with modified parameters
                modified_params = self._modify_params(best_overall_params)
                recursive_params, recursive_output, recursive_prompt = self._optimize_recursive(
                    best_overall_prompt, reference, method, n_samples, recursion_depth + 1
                )
                recursive_score = self.scorer.score(recursive_output, reference)

                if recursive_score > best_overall_score:
                    best_overall_params = recursive_params
                    best_overall_output = recursive_output
                    best_overall_score = recursive_score
                    best_overall_prompt = recursive_prompt
                else:
                    # If recursive call didn't improve, break the loop
                    print("No improvement in this iteration and recursive call. Stopping.")
                    break

            self.meta_predictor.train(extract_prompt_features(best_overall_prompt), best_overall_params)

        return best_overall_params, best_overall_output, best_overall_prompt

    def _optimize_iterative(self, prompt: str, reference: str, method: str, n_samples: int) -> Tuple[Dict[str, float], str, float]:
        def score_function(output):
            return self.scorer.score(output, reference)

        def generate_and_score(prompt, **params):
            output = self.llm_wrapper.generate(prompt, **params)
            return output, score_function(output) if output is not None else (None, float('-inf'))

        initial_params = self.meta_predictor.predict(extract_prompt_features(prompt))
        initial_output, initial_score = generate_and_score(prompt, **initial_params)
        results = [(initial_params, initial_output, initial_score)]

        explored_results = self.explorer.explore(lambda p, **params: generate_and_score(prompt, **params), 
                                                 prompt, method, n_samples)
        results.extend([(params, output, score) for params, (output, score) in explored_results])
        
        best_params, best_output, best_score = max(results, key=lambda x: x[2])

        return best_params, best_output, best_score

    def _modify_params(self, params: Dict[str, float]) -> Dict[str, float]:
        modified_params = params.copy()
        param_to_modify = random.choice(list(params.keys()))
        modification = random.uniform(-0.1, 0.1)  # Modify by up to ±10%
        modified_params[param_to_modify] *= (1 + modification)
        return modified_params

    def parameter_importance(self, results: List[Tuple[Dict[str, float], str, float]]) -> Dict[str, float]:
        X = np.array([[p[k] for k in p.keys()] for p, _, _ in results])
        y = np.array([score for _, _, score in results])
        
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance = {k: v for k, v in zip(results[0][0].keys(), rf.feature_importances_)}
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
