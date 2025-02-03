!pip install transformers torch emoji scikit-learn pandas rouge_score
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from rouge_score import rouge_scorer
import re
import unicodedata
import emoji
import os
import pandas as pd

# Path to the uploaded file in Colab
file_path = '/content/CozmoX Assignment Dataset.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Check if it loaded correctly
print(df.head())






# Columns: 'input_text' for UK Dialect and 'target_text' for US Dialect
input_texts = df['input_text'].tolist()
target_texts = df['target_text'].tolist()

# Split the data into training and testing sets (80/20 split)
train_inputs, test_inputs, train_targets, test_targets = train_test_split(input_texts, target_texts, test_size=0.2, random_state=42)

# 2. Model Selection & Justification
class DialectProcessor:
    """Handles text preprocessing and UK to US conversion"""
    def __init__(self):
        self.uk_to_us_mappings = {
            'theatre': 'theater', 'colour': 'color', 'flavour': 'flavor', 'centre': 'center',
            'metre': 'meter', 'litre': 'liter', 'catalogue': 'catalog', 'cheque': 'check',
            'defence': 'defense', 'licence': 'license', 'practise': 'practice', 'offence': 'offense',
            'pretence': 'pretense', 'flat': 'apartment', 'lift': 'elevator', 'football': 'soccer',
            'aeroplane': 'airplane', 'jewellery': 'jewelry', 'programme': 'program', 'labour': 'labor',
            'neighbour': 'neighbor', 'behaviour': 'behavior', 'favourite': 'favorite',
            'travelling': 'traveling', 'cancelled': 'canceled', 'spanner': 'wrench'
        }

    def preprocess_text(self, text: str) -> str:
        """Normalizes and cleans input text"""
        text = unicodedata.normalize('NFKC', text).strip()
        text = self._convert_emojis_to_placeholders(text)
        text = re.sub(r'\s+', ' ', text)
        text = text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII
        return text

    def convert_dialect(self, text: str) -> str:
        """Converts UK English words to US English using manual mappings"""
        words = text.split()
        converted_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in self.uk_to_us_mappings:
                converted_words.append(self.uk_to_us_mappings[lower_word])
            else:
                converted_words.append(word)
        return " ".join(converted_words)

    def _convert_emojis_to_placeholders(self, text: str) -> str:
        """Handles emoji conversion"""
        emoji_pattern = r'[\U00010000-\U0010ffff]|[\u2000-\u2bff]|\u25aa|\u25ab|\u2300-\u23ff|\u2b50'
        return re.sub(emoji_pattern, lambda match: f" <emoji>{emoji.demojize(match.group(0))}</emoji> ", text)


# 3. Model Selection
class DialectConverter:
    """Main class for dialect conversion model"""
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer, self.model = self.load_model(model_name)
        self.processor = DialectProcessor()

    def load_model(self, model_name: str):
        """Load a specific pre-trained model and tokenizer."""
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        return tokenizer, model

    def infer(self, text: str) -> str:
        """Perform inference using model and rule-based conversion"""
        preprocessed_text = self.processor.preprocess_text(text)
        rule_based_conversion = self.processor.convert_dialect(preprocessed_text)

        input_text = f"Translate the following UK English to US English: {rule_based_conversion}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=128)

        model_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if len(model_output.split()) >= len(rule_based_conversion.split()):
            return model_output
        return rule_based_conversion


# 4. Model Training & Evaluation
def evaluate_model(converter, test_inputs, test_targets):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    total_count = len(test_inputs)

    for i, sentence in enumerate(test_inputs):
        converted_text = converter.infer(sentence)
        expected_text = test_targets[i]

        scores = scorer.score(expected_text, converted_text)

        total_rouge1 += scores['rouge1'].fmeasure
        total_rouge2 += scores['rouge2'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure

        # Optionally, print examples for debugging
        print(f"Input: {sentence}\nExpected: {expected_text}\nConverted: {converted_text}\n{'-'*50}")

    avg_rouge1 = total_rouge1 / total_count
    avg_rouge2 = total_rouge2 / total_count
    avg_rougeL = total_rougeL / total_count

    print(f"Average ROUGE-1: {avg_rouge1:.2f}")
    print(f"Average ROUGE-2: {avg_rouge2:.2f}")
    print(f"Average ROUGE-L: {avg_rougeL:.2f}")

    return avg_rouge1, avg_rouge2, avg_rougeL


# 5. Deployment & Inference
# Run the evaluation
converter = DialectConverter(model_name="facebook/bart-large-mnli")

# Evaluate the model with the test data (80/20 split)
evaluate_model(converter, test_inputs, test_targets)
