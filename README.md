## Installation and Setup

To install and set up the LLM Visualization Tool, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/llm-visualization-tool.git
    cd llm-visualization-tool
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

## Usage Examples

### Analyzing Text with LLMAnalyzer

```python
from llm_analyzer import LLMAnalyzer

# Initialize the analyzer with a model
analyzer = LLMAnalyzer(model_name='gpt-2')

# Process text
analysis_results = analyzer.process_text("Sample text for analysis")

# Generate text
generated_text = analyzer.generate_text("Starting prompt")

# Calculate feature importance
feature_importance = analyzer.calculate_feature_importance("Sample text for analysis")

# Perform behavioral clustering
clustering_results = analyzer.cluster_behavioral_patterns(["Text sample 1", "Text sample 2"])
```

### Creating Visualizations

```python
from visualization import create_animated_tokenization, evaluate_model

# Create an animated tokenization visualization
create_animated_tokenization("Sample text for tokenization")

# Evaluate model performance
performance_metrics = evaluate_model("Sample text for evaluation")
```

## Contribution Guidelines

We welcome contributions to the LLM Visualization Tool! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

Please ensure your code adheres to our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


# LLM Visualization Tool Documentation

## Overview
A Streamlit-based web application for analyzing and visualizing Large Language Models (LLMs). This tool provides interactive visualizations and analysis of model behavior, including tokenization, attention patterns, and behavioral clustering.

## Key Features
- Dynamic model loading from HuggingFace
- Interactive tokenization visualization
- Attention pattern analysis
- Text generation capabilities
- Advanced model behavior analysis
- Export functionality for analysis reports

## Components

### Model Support
- GPT-2
- BERT
- T5
- LLaMA
- Custom models from HuggingFace

### Main Classes

#### LLMAnalyzer
Primary class for model analysis and processing.

**Key Methods:**
- `process_text()`: Analyzes input text and returns tokenization, attentions, and hidden states
- `generate_text()`: Generates new text sequences
- `calculate_feature_importance()`: Evaluates token importance
- `cluster_behavioral_patterns()`: Performs behavioral clustering analysis

### Visualization Functions

#### create_animated_tokenization()
Creates an animated visualization of the tokenization process using Plotly.

#### evaluate_model()
Calculates model performance metrics:
- Perplexity
- Token probability
- Loss

### UI Components

#### Sidebar
- Model selection
- Configuration options
- Device information

#### Main Tabs
1. **Tokenization & Probabilities**
    - Animated tokenization process
    - Token distribution visualization

2. **Attention Patterns**
    - Interactive attention matrix visualization
    - Layer and head selection

3. **Model Analysis**
    - Performance metrics
    - Statistical analysis

4. **Generated Output**
    - Text generation results
    - Token-by-token breakdown

5. **Advanced Analysis**
    - Feature importance visualization
    - Behavioral clustering
    - Analysis export functionality

## Technical Requirements
- Python 3.6+
- PyTorch
- Streamlit
- Transformers library
- CUDA-capable GPU (optional)

## Error Handling
- Comprehensive error handling for model loading
- Validation of model names
- Process monitoring and logging

## Data Export
Supports exporting analysis results in JSON format, including:
- Text analysis
- Performance metrics
- Token processing
- Attention patterns
- Hidden states
