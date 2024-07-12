# dataset_loader.py

from datasets import load_dataset
from typing import List, Dict
import random

class OpenAssistantDatasetLoader(DatasetLoader):
    def __init__(self, local_file: str = 'openassistant_local_dataset.json'):
        super().__init__('OpenAssistant/oasst1', None, local_file)

    def format_entry(self, entry: Dict, source: str) -> Dict:
        if source == 'huggingface':
            # Find the first human query and the following assistant response
            human_query = ""
            assistant_response = ""
            for message in entry['messages']:
                if message['role'] == 'prompter' and not human_query:
                    human_query = message['text']
                elif message['role'] == 'assistant' and human_query and not assistant_response:
                    assistant_response = message['text']
                    break

            return {
                'prompt': human_query,
                'reference': assistant_response,
                'params': {},  # Initialize with empty params as they're not provided
                'output': "",  # Initialize with empty output
                'score': 0.0  # Initialize with a placeholder score
            }
        return entry  # Local entries are already in the correct format
