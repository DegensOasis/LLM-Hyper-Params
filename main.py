from llm_param_optimizer.explorer import ParameterExplorer
from llm_param_optimizer.scorer import Scorer, length_score, diversity_score, relevance_score
from llm_param_optimizer.optimizer import Optimizer
from llm_param_optimizer.llm_wrapper import OpenAIWrapper, HuggingFaceWrapper
from llm_param_optimizer.models.meta_predictor import MetaPredictor
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
    
    # Choose your LLM wrapper here
    llm_wrapper = OpenAIWrapper(api_key=os.getenv("OPENAI_API_KEY"))
    # llm_wrapper = HuggingFaceWrapper(model_name="gpt2")  # or any other model name
    
    meta_predictor = MetaPredictor()

    optimizer = Optimizer(explorer, scorer, llm_wrapper, meta_predictor, max_iterations=5, max_recursion_depth=3)

    original_prompt = "Explain machine learning."
    reference = "Machine learning is a type of artificial intelligence that allows computer systems to automatically learn and improve from experience without being explicitly programmed."

    best_params, best_output, best_prompt = optimizer.optimize(original_prompt, reference, method='random', n_samples=20)

    print(f"\nFinal Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best prompt: {best_prompt}")
    print(f"Best output: {best_output}")

    # Analyze parameter importance
    all_results = optimizer.explorer.explore(lambda p, **params: (optimizer.llm_wrapper.generate(best_prompt, **params),
                                                                  optimizer.scorer.score(optimizer.llm_wrapper.generate(best_prompt, **params), reference)),
                                             best_prompt, method='random', n_samples=100)
    importance = optimizer.parameter_importance(all_results)
    print("\nParameter Importance:")
    for param, imp in importance.items():
        print(f"{param}: {imp:.4f}")

    meta_predictor.save('meta_predictor.joblib')

if __name__ == "__main__":
    main()
