# LLM-Hyper-Params
A WIP open sourced Python wrapper for LLM parameter optimization and queries.

## Features

1. **Universal LLM Compatibility**
   - Works with various Language Model providers (OpenAI, Hugging Face, etc.)
   - Easily extendable to support new LLM APIs

2. **Advanced Parameter Optimization**
   - Supports multiple optimization strategies:
     - Grid Search
     - Random Search
   - Iterative and recursive optimization process
   - Adaptive parameter modification

3. **Prompt Optimization**
   - Automatic prompt rephrasing for improved results
   - Combines parameter and prompt optimization

4. **Sophisticated Scoring System**
   - Customizable scoring functions
   - Built-in scoring for length, diversity, and relevance
   - Easy to add new scoring criteria

5. **Meta-Learning Capabilities**
   - Learns from previous optimizations to predict good starting parameters
   - Saves and loads learned meta-predictor models

6. **Parameter Importance Analysis**
   - Provides insights into which parameters have the most impact on output quality

7. **Flexible Configuration**
   - Customizable parameter ranges
   - Adjustable number of iterations and recursion depth

8. **Comprehensive Logging**
   - Detailed output of optimization process
   - Final results summary with best parameters, prompt, and output

## FAQ

1. **Q: What LLMs does this optimizer support?**
   A: The optimizer is designed to be universal. It currently includes wrappers for OpenAI and Hugging Face models, but can be easily extended to support other LLM providers.

2. **Q: How does the optimizer improve prompts?**
   A: The optimizer uses the LLM itself to rephrase the original prompt, aiming to make it clearer and more specific. This process is repeated iteratively along with parameter optimization.

3. **Q: Can I add my own scoring functions?**
   A: Yes, you can easily add custom scoring functions by defining them and including them in the `Scorer` initialization in `main.py`.

4. **Q: What does the meta-predictor do?**
   A: The meta-predictor learns from previous optimizations to predict good starting parameters for new prompts. This can speed up the optimization process over time.

5. **Q: How do I use this with a different LLM provider?**
   A: To use a new LLM provider, create a new wrapper class in `llm_wrapper.py` that inherits from `LLMWrapper` and implements the `generate` method. Then, instantiate this new wrapper in `main.py`.

6. **Q: What does the parameter importance analysis tell me?**
   A: The parameter importance analysis shows which parameters have the most significant impact on the output quality. This can help in focusing optimization efforts and understanding the LLM's behavior.

7. **Q: Can this be used for tasks other than text generation?**
   A: While the current implementation is focused on text generation, the core optimization logic could be adapted for other LLM tasks by modifying the scoring functions and output processing.

8. **Q: How do I install and run this optimizer?**
   A: Clone the repository, install the required dependencies (listed in `requirements.txt`), set up your LLM API credentials, and run `main.py`. Detailed installation instructions should be included in the project's README file.

9. **Q: Is this optimizer suitable for production use?**
   A: While the optimizer can provide valuable insights and improvements, it's primarily designed as a research and development tool. For production use, thorough testing and potentially some optimizations for efficiency would be recommended.

10. **Q: How can I contribute to this project?**
    A: Contributions are welcome! You could add support for new LLM providers, implement new optimization strategies, create additional scoring functions, or improve the documentation.
