import os
import json

class DataLoader:
    """Handles secure filesystem reads and data validation for the matching engine."""
    
    @staticmethod
    def load_json(file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required input file asset at: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON schema discovered in {file_path}: {str(e)}")