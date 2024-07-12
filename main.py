# main.py

import os
import argparse
import json
from dotenv import load_dotenv
from llm_param_optimizer.explorer import ParameterExplorer
from llm_param_optimizer.scorer import Scorer, length_score, diversity_score, relevance_score
from llm_param_optimizer.optimizer import Optimizer
from llm_param_optimizer.llm_wrapper import OpenAIWrapper, HuggingFaceWrapper
from llm_param_optimizer.models.meta_predictor import MetaPredictor
from dataset_loader import OpenAssistantDatasetLoader

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Parameter Optimizer")
    parser.add_argument("--llm", choices=["openai", "huggingface"], default="openai", help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name (for Hugging Face)")
    parser.add_argument("--method", choices=["grid", "random"], default="random", help="Search method")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples for random search")
    parser.add_argument("--iterations", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("--recursion-depth", type=int, default=3, help="Maximum recursion depth")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts to optimize")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    return parser.parse_args()

def main():
    args = parse_args()

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
    
    if args.llm == "openai":
        llm_wrapper = OpenAIWrapper(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        llm_wrapper = HuggingFaceWrapper(model_name=args.model)
    
    meta_predictor = MetaPredictor()

    optimizer = Optimizer(explorer, scorer, llm_wrapper, meta_predictor, 
                          max_iterations=args.iterations, max_recursion_depth=args.recursion_depth)

    # Load dataset
    dataset_loader = OpenAssistantDatasetLoader()
    
    # Get multiple entries for optimization
    entries = dataset_loader.get_entries(args.num_prompts)

    results = []

    for entry in entries:
        print(f"\nOptimizing for prompt: {entry['prompt']}")
        
        try:
            best_params, best_output, best_prompt = optimizer.optimize(
                entry['prompt'], 
                entry['reference'], 
                method=args.method, 
                n_samples=args.samples
            )

            print(f"Best parameters: {best_params}")
            print(f"Best prompt: {best_prompt}")
            print(f"Best output: {best_output}")

            # Update the local dataset with the optimized result
            score = scorer.score(best_output, entry['reference'])
            dataset_loader.add_local_entry(best_prompt, entry['reference'], best_params, best_output, score)

            # Analyze parameter importance
            all_results = optimizer.explorer.explore(
                lambda p, **params: (
                    optimizer.llm_wrapper.generate(best_prompt, **params),
                    optimizer.scorer.score(optimizer.llm_wrapper.generate(best_prompt, **params), entry['reference'])
                ),
                best_prompt, 
                method='random', 
                n_samples=100
            )
            importance = optimizer.parameter_importance(all_results)
            print("\nParameter Importance:")
            for param, imp in importance.items():
                print(f"{param}: {imp:.4f}")

            results.append({
                "original_prompt": entry['prompt'],
                "optimized_prompt": best_prompt,
                "best_params": best_params,
                "best_output": best_output,
                "score": score,
                "parameter_importance": importance
            })

        except Exception as e:
            print(f"Error occurred while optimizing prompt: {str(e)}")
            continue

    # Save the updated local dataset
    dataset_loader.save_local_dataset('openassistant_local_dataset.json')

    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    meta_predictor.save('meta_predictor.joblib')

if __name__ == "__main__":
    main()
