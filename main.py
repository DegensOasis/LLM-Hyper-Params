# main.py
from llm_param_optimizer.explorer import ParameterExplorer
from llm_param_optimizer.scorer import Scorer, length_score, diversity_score, relevance_score
from llm_param_optimizer.optimizer import Optimizer
from llm_param_optimizer.llm_wrapper import OpenAIWrapper, HuggingFaceWrapper
from llm_param_optimizer.models.meta_predictor import MetaPredictor
from dataset_loader import ELI5DatasetLoader
import os

def main():
    param_ranges = {
        'temperature': (0.1, 2.0, 0.1),
        'top_p': (0.1, 1.0, 0.1),
        'top_k': (1, 100, 1),
        'frequency_penalty': (0.0, 2.0, 0.1),
        'presence_penalty': (0.0, 2.0, 0.1),
        'repetition_penalty': (1.0, 2.0, 0.1),
    }

    explorer = ParameterExplorer(param_ranges)
    scorer = Scorer([length_score, diversity_score, relevance_score])
    
    llm_wrapper = OpenAIWrapper(api_key=os.getenv("OPENAI_API_KEY"))
    
    meta_predictor = MetaPredictor()

    optimizer = Optimizer(explorer, scorer, llm_wrapper, meta_predictor, max_iterations=5, max_recursion_depth=3)

    # Load dataset
    dataset_loader = ELI5DatasetLoader()
    
    # Get multiple entries for optimization
    entries = dataset_loader.get_entries(5)  # Get 5 entries, mixing Hugging Face and local data if available

    for entry in entries:
        print(f"\nOptimizing for prompt: {entry['prompt']}")
        best_params, best_output, best_prompt = optimizer.optimize(entry['prompt'], entry['reference'], method='random', n_samples=20)

        print(f"Best parameters: {best_params}")
        print(f"Best prompt: {best_prompt}")
        print(f"Best output: {best_output}")

        # Update the local dataset with the optimized result
        dataset_loader.add_local_entry(best_prompt, entry['reference'], best_params, best_output, scorer.score(best_output, entry['reference']))

        # Analyze parameter importance
        all_results = optimizer.explorer.explore(lambda p, **params: (optimizer.llm_wrapper.generate(best_prompt, **params),
                                                                      optimizer.scorer.score(optimizer.llm_wrapper.generate(best_prompt, **params), entry['reference'])),
                                                 best_prompt, method='random', n_samples=100)
        importance = optimizer.parameter_importance(all_results)
        print("\nParameter Importance:")
        for param, imp in importance.items():
            print(f"{param}: {imp:.4f}")

    # Save the updated local dataset
    dataset_loader.save_local_dataset('eli5_local_dataset.json')

    meta_predictor.save('meta_predictor.joblib')

if __name__ == "__main__":
    main()
