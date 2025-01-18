# LLM Visualization Tool

This project is a dynamic visualization tool for analyzing and understanding the behavior of Large Language Models (LLMs) using Streamlit. It allows users to load various pre-trained models from HuggingFace, process text inputs, visualize tokenization, attention patterns, and generate text sequences.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Technical Documentation](#technical-documentation)
    - [Model Initialization](#model-initialization)
    - [Text Processing](#text-processing)
    - [Text Generation](#text-generation)
    - [Feature Importance Calculation](#feature-importance-calculation)
    - [Behavioral Clustering](#behavioral-clustering)
    - [Visualization](#visualization)
    - [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Features

- Load various pre-trained models from HuggingFace.
- Process and analyze text inputs.
- Visualize tokenization and attention patterns.
- Generate text sequences.
- Calculate feature importance based on attention weights.
- Cluster behavioral patterns using hidden states.
- Export analysis reports.

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

```shell
git clone https://github.com/sidharthsajith/LLM-Analyser.git
cd LLM-Analyser
```

2. Install the required packages:

```shell
pip install streamlit transformers huggingface_hub scikit-learn torch plotly pandas numpy
```

3. Run the Streamlit app:

```shell
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Use the sidebar to select or enter a HuggingFace model name.
3. Load the model.
4. Enter text to analyze in the main content area.
5. Click "Analyze" to process the text and visualize the results.
6. Explore the various tabs for tokenization, attention patterns, model analysis, generated output, and advanced analysis.

## Technical Documentation

### Model Initialization

The `LLMAnalyzer` class is responsible for initializing the model and tokenizer based on the selected model name. It supports various architectures such as GPT-2, BERT, T5, and LLaMA.

```python
class LLMAnalyzer:
        def __init__(self, model_name: str = None, model=None, tokenizer=None):
                # Initialization code
```

### Text Processing

The `process_text` method processes the input text and returns tokenization, attentions, and hidden states.

```python
def process_text(self, text: str):
        # Text processing code
```

### Text Generation

The `generate_text` method generates text sequences based on the input IDs.

```python
def generate_text(self, input_ids, num_sequences=3):
        # Text generation code
```

### Feature Importance Calculation

The `calculate_feature_importance` method calculates feature importance based on attention weights.

```python
def calculate_feature_importance(self, attentions):
        # Feature importance calculation code
```

### Behavioral Clustering

The `cluster_behavioral_patterns` method clusters behavioral patterns using hidden states.

```python
def cluster_behavioral_patterns(self, hidden_states, n_clusters=3):
        # Behavioral clustering code
```

### Visualization

The `create_animated_tokenization` function creates an animated visualization of the tokenization process.

```python
def create_animated_tokenization(token_process, auto_play=True):
        # Visualization code
```

### Model Evaluation

The `evaluate_model` function evaluates model performance metrics such as perplexity, average token probability, and loss.

```python
def evaluate_model(analyzer, text):
        # Model evaluation code
```
