from abc import ABC, abstractmethod
from .config import DEFAULT_PARAMS

class LLMWrapper(ABC):
    def __init__(self):
        self.default_params = DEFAULT_PARAMS.copy()

    @abstractmethod
    def generate(self, prompt: str, **params):
        pass

class OpenAIWrapper(LLMWrapper):
    def __init__(self, api_key):
        super().__init__()
        import openai
        openai.api_key = api_key
        self.client = openai.ChatCompletion

    def generate(self, prompt: str, **params):
        try:
            call_params = self.default_params.copy()
            call_params.update(params)
            response = self.client.create(
                model=call_params.pop('model', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                **call_params
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return None

class HuggingFaceWrapper(LLMWrapper):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt: str, **params):
        try:
            call_params = self.default_params.copy()
            call_params.update(params)
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, **call_params)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error in Hugging Face model generation: {str(e)}")
            return None

# Add more wrapper classes for other LLM providers as needed
