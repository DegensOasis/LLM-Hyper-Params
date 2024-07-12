from typing import List, Callable
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt', quiet=True)

class Scorer:
    def __init__(self, scoring_functions: List[Callable]):
        self.scoring_functions = scoring_functions
    
    def score(self, output: str, reference: str = None) -> float:
        scores = [func(output, reference) for func in self.scoring_functions]
        return sum(scores) / len(scores)

def length_score(output: str, reference: str = None) -> float:
    target_length = 100  # Adjust as needed
    output_length = len(output.split())
    return max(0, 1 - abs(output_length - target_length) / target_length)

def diversity_score(output: str, reference: str = None) -> float:
    tokens = word_tokenize(output.lower())
    return len(set(tokens)) / len(tokens) if tokens else 0

def relevance_score(output: str, reference: str) -> float:
    reference_tokens = [word_tokenize(reference.lower())]
    output_tokens = word_tokenize(output.lower())
    return sentence_bleu(reference_tokens, output_tokens)
