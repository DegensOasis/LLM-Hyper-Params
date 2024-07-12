# dataset_loader.py

from datasets import load_dataset
from typing import List, Dict
import random
import json
import os

class Dataset:
    def __init__(self):
        self.entries: List[Dict] = []

    def add_entry(self, prompt: str, reference: str, params: Dict[str, float], output: str, score: float):
        self.entries.append({
            'prompt': prompt,
            'reference': reference,
            'params': params,
            'output': output,
            'score': score
        })

    def save(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.entries, f, indent=2)

    @classmethod
    def load(cls, filename: str):
        dataset = cls()
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                dataset.entries = json.load(f)
        return dataset

class DatasetLoader:
    def __init__(self, dataset_name: str = None, subset: str = None, local_file: str = None):
        self.huggingface_dataset = None
        self.local_dataset = None
        self.current_split = 'train'

        if dataset_name:
            self.huggingface_dataset = load_dataset(dataset_name, subset)
        
        if local_file:
            self.local_dataset = Dataset.load(local_file)

    def get_entry(self, source: str = 'any') -> Dict:
        if source == 'huggingface' and self.huggingface_dataset:
            entry = random.choice(self.huggingface_dataset[self.current_split])
            return self.format_entry(entry, 'huggingface')
        elif source == 'local' and self.local_dataset:
            return random.choice(self.local_dataset.entries)
        elif source == 'any':
            if self.huggingface_dataset and self.local_dataset:
                return random.choice([self.get_entry('huggingface'), self.get_entry('local')])
            elif self.huggingface_dataset:
                return self.get_entry('huggingface')
            elif self.local_dataset:
                return self.get_entry('local')
        
        raise ValueError("No dataset available for the specified source")

    def get_entries(self, n: int, source: str = 'any') -> List[Dict]:
        return [self.get_entry(source) for _ in range(n)]

    def format_entry(self, entry: Dict, source: str) -> Dict:
        if source == 'huggingface':
            # Override this method in subclasses to format specific Hugging Face datasets
            return entry
        return entry

    def set_split(self, split: str):
        if self.huggingface_dataset and split in self.huggingface_dataset.keys():
            self.current_split = split
        else:
            raise ValueError(f"Split {split} not found in dataset")

    def add_local_entry(self, prompt: str, reference: str, params: Dict[str, float], output: str, score: float):
        if not self.local_dataset:
            self.local_dataset = Dataset()
        self.local_dataset.add_entry(prompt, reference, params, output, score)

    def save_local_dataset(self, filename: str):
        if self.local_dataset:
            self.local_dataset.save(filename)

class ELI5DatasetLoader(DatasetLoader):
    def __init__(self, local_file: str = 'eli5_local_dataset.json'):
        super().__init__('eli5', 'arxiv', local_file)

    def format_entry(self, entry: Dict, source: str) -> Dict:
        if source == 'huggingface':
            return {
                'prompt': entry['title'],
                'reference': entry['answers']['text'][0] if entry['answers']['text'] else "",
                'params': {},  # Initialize with empty params
                'output': "",  # Initialize with empty output
                'score': 0.0  # Initialize with a placeholder score
            }
        return entry  # Local entries are already in the correct format

# Add more dataset loaders for different Hugging Face datasets as needed
