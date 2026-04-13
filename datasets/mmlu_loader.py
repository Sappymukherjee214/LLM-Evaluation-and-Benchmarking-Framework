import pandas as pd
import json
import os
from typing import List
from .base_dataset import DatasetItem

class MMLULoader:
    """
    Loader for the MMLU dataset from Kaggle.
    Converts CSV into the framework's internal format.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def convert(self, limit_per_subject: int = 5) -> List[DatasetItem]:
        df = pd.read_csv(self.csv_path)
        items = []
        
        # Group by subject to get a balanced sample if needed
        subjects = df['Subject'].unique()
        
        for subject in subjects:
            subj_df = df[df['Subject'] == subject].head(limit_per_subject)
            for idx, row in subj_df.iterrows():
                # Format the question with A, B, C, D options
                prompt = f"Subject: {row['Subject']}\n\n"
                prompt += f"Question: {row['Question']}\n"
                prompt += f"A) {row['A']}\n"
                prompt += f"B) {row['B']}\n"
                prompt += f"C) {row['C']}\n"
                prompt += f"D) {row['D']}\n"
                prompt += "\nAnswer with only the letter of the correct option (A, B, C, or D)."
                
                items.append(DatasetItem(
                    id=f"mmlu_{subject}_{idx}",
                    input=prompt,
                    expected_output=str(row['Answer']).strip(),
                    metadata={
                        "subject": row['Subject'],
                        "source": "MMLU",
                        "type": "multiple_choice"
                    }
                ))
        
        return items

    def save_to_json(self, output_path: str, limit_per_subject: int = 5):
        items = self.convert(limit_per_subject)
        data = {
            "name": "MMLU Benchmark (Subsampled)",
            "version": "1.0.0",
            "items": [item.model_dump() for item in items]
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"MMLU dataset converted and saved to {output_path}. Total items: {len(items)}")

if __name__ == "__main__":
    # This is for manual testing of the loader
    import sys
    if len(sys.argv) > 1:
        loader = MMLULoader(sys.argv[1])
        loader.save_to_json("datasets/mmlu_sample.json", limit_per_subject=2)
