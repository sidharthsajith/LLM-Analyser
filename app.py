import torch
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    BertTokenizerFast,
    BertForMaskedLM,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoConfig,
)
import numpy as np
import pandas as pd
from huggingface_hub import model_info
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="LLM Visualization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2E86C1;
    }
    .stProgress > div > div > div {
        background-color: #2E86C1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session states
if "analyzer" not in st.session_state:
    st.session_state["analyzer"] = None
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None
if "generation_complete" not in st.session_state:
    st.session_state["generation_complete"] = False

def validate_model_name(model_name: str) -> bool:
    """Validate if the model exists on HuggingFace."""
    try:
        info = model_info(model_name)
        return True
    except Exception as e:
        logger.warning(f"Model validation failed: {str(e)}")
        return False

def get_model_architecture(model_name: str) -> str:
    """Determine the model architecture type using its configuration."""
    try:
        model_config = AutoConfig.from_pretrained(model_name)
        if "gpt2" in model_config.architectures[0].lower():
            return "gpt2"
        elif "bert" in model_config.architectures[0].lower():
            return "bert"
        elif "t5" in model_config.architectures[0].lower():
            return "t5"
        elif "llama" in model_config.architectures[0].lower():
            return "llama"
        else:
            return "unknown"
    except Exception as e:
        logger.warning(f"Architecture detection failed: {str(e)}")
        return "unknown"

class LLMAnalyzer:
    def __init__(self, model_name: str = None, model=None, tokenizer=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model and tokenizer:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        else:
            self.model_name = model_name or "gpt2"
            try:
                self.architecture = get_model_architecture(self.model_name)
                logger.info(f"Detected architecture: {self.architecture}")

                if self.architecture == "gpt2":
                    self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
                    self.model = GPT2LMHeadModel.from_pretrained(
                        self.model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                elif self.architecture == "bert":
                    self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
                    self.model = BertForMaskedLM.from_pretrained(
                        self.model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                elif self.architecture == "t5":
                    self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                elif self.architecture == "llama":
                    self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
                    self.model = LlamaForCausalLM.from_pretrained(
                        self.model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        output_attentions=True,
                        output_hidden_states=True,
                    )

                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = self.model.to(self.device)

            except Exception as e:
                logger.error(f"Model initialization failed: {str(e)}")
                st.error(f"Failed to initialize model: {str(e)}")
                raise e

        self.config = {
            "num_layers": self.model.config.num_hidden_layers,
            "num_heads": self.model.config.num_attention_heads,
            "hidden_size": self.model.config.hidden_size,
            "architecture": self.architecture,
        }

    def process_text(self, text: str):
        """Process input text and return tokenization, attentions, and hidden states."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            token_process = []
            for i, token_id in enumerate(inputs["input_ids"][0]):
                token = self.tokenizer.decode(token_id)
                token_process.append({"step": i, "token": token, "token_id": token_id.item()})

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            if outputs is None:
                raise ValueError("Model returned no outputs. Check the input text and model configuration.")

            attentions = [att.cpu() for att in outputs.attentions] if outputs.attentions else None
            hidden_states = [hs.cpu() for hs in outputs.hidden_states] if outputs.hidden_states else None
            logits = outputs.logits.cpu() if outputs.logits is not None else None
            loss = outputs.loss.cpu() if hasattr(outputs, "loss") and outputs.loss is not None else None

            return {
                "token_process": token_process,
                "attentions": attentions,
                "hidden_states": hidden_states,
                "logits": logits,
                "input_ids": inputs["input_ids"].cpu(),
                "loss": loss,
            }
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            st.error(f"An error occurred during text processing: {str(e)}")
            return None

    def generate_text(self, input_ids, num_sequences=3):
        """Generate text sequences with proper error handling."""
        try:
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=min(100, self.model.config.max_position_embeddings),
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    num_return_sequences=num_sequences,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )

            sequences = []
            for seq in generated.sequences:
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                tokens = self.tokenizer.convert_ids_to_tokens(seq)
                sequences.append({"text": decoded, "tokens": tokens})

            return sequences
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            st.error(f"An error occurred during text generation: {str(e)}")
            return None

    def calculate_feature_importance(self, attentions):
        """Calculate feature importance based on attention weights."""
        if attentions is None:
            return None

        # Aggregate attention weights across layers and heads
        aggregated_attention = torch.stack(attentions).mean(dim=(0, 2))  # Average over layers and heads
        return aggregated_attention.numpy()

    def cluster_behavioral_patterns(self, hidden_states, n_clusters=3):
        """Cluster behavioral patterns using hidden states."""
        if hidden_states is None:
            return None

        # Flatten hidden states for clustering
        hidden_states_flat = torch.cat(hidden_states, dim=1).squeeze(0).numpy()
        pca = PCA(n_components=2)
        reduced_states = pca.fit_transform(hidden_states_flat)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(reduced_states)

        return reduced_states, clusters

def create_animated_tokenization(token_process, auto_play=True):
    """Create an animated visualization of the tokenization process."""
    frames = []
    max_steps = len(token_process)

    for i in range(max_steps):
        df = pd.DataFrame(token_process[: i + 1])
        frame = go.Frame(
            data=[
                go.Bar(
                    x=df["step"],
                    y=[1] * len(df),
                    text=df["token"],
                    textposition="auto",
                    name="Tokens",
                    marker_color="lightblue",
                )
            ],
            name=f"frame{i}",
        )
        frames.append(frame)

    fig = go.Figure(
        data=[go.Bar(x=[0], y=[1], text=[""], textposition="auto")],
        frames=frames,
    )

    fig.update_layout(
        title="Tokenization Process",
        xaxis_title="Token Position",
        yaxis_showticklabels=False,
        height=400,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Token: ", "visible": True},
                "steps": [
                    {
                        "args": [[f"frame{k}"], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k in range(max_steps)
                ],
            }
        ],
    )

    return fig

def evaluate_model(analyzer, text):
    """Evaluate model performance metrics with proper error handling."""
    try:
        inputs = analyzer.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(analyzer.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = analyzer.model(**inputs)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss.item()
            else:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

            perplexity = np.exp(loss)

            probs = torch.nn.functional.softmax(outputs.logits[0].cpu(), dim=-1)
            token_probs = torch.gather(probs[:-1], 1, inputs["input_ids"][0, 1:].cpu().unsqueeze(-1)).squeeze(-1)
            avg_token_prob = token_probs.mean().item()

            return {
                "perplexity": perplexity,
                "avg_token_prob": avg_token_prob,
                "loss": loss,
            }
    except Exception as e:
        logger.warning(f"Error calculating metrics: {str(e)}")
        return {
            "perplexity": float("nan"),
            "avg_token_prob": float("nan"),
            "loss": float("nan"),
        }

def download_link(object_to_download, download_filename, download_link_text):
    """Generates a link to download the given object_to_download."""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Streamlit UI
st.title("Dynamic LLM Analysis Tool")

# Sidebar for model selection and configuration
with st.sidebar:
    st.header("Model Configuration")

    custom_model = st.text_input(
        "Enter HuggingFace model name (optional)",
        placeholder="e.g., gpt2-medium or your-username/your-model",
    )

    if custom_model:
        if validate_model_name(custom_model):
            model_options = {
                "Custom Model": custom_model,
                "GPT-2": "gpt2",
                "GPT-2 Medium": "gpt2-medium",
                "BERT Base": "bert-base-uncased",
                "RoBERTa Base": "roberta-base",
            }
            st.success("✅ Valid model found on HuggingFace!")
        else:
            st.error("❌ Model not found on HuggingFace")
            model_options = {
                "GPT-2": "gpt2",
                "GPT-2 Medium": "gpt2-medium",
                "BERT Base": "bert-base-uncased",
                "RoBERTa Base": "roberta-base",
            }
    else:
        model_options = {
            "GPT-2": "gpt2",
            "GPT-2 Medium": "gpt2-medium",
            "BERT Base": "bert-base-uncased",
            "RoBERTa Base": "roberta-base",
        }

    selected_model = st.selectbox("Select Model", list(model_options.keys()), key="model_selector")

    if st.button("Load Model"):
        try:
            with st.spinner("Loading model..."):
                st.session_state["analyzer"] = LLMAnalyzer(model_name=model_options[selected_model])
                st.session_state["model_loaded"] = True
                st.success(f"{selected_model} loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.session_state["model_loaded"] = False
            logger.error(f"Model loading failed: {str(e)}")

    st.info(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Main content area
if st.session_state["model_loaded"] and st.session_state["analyzer"] is not None:
    text = st.text_area(
        "Enter text to analyze:",
        "The quick brown fox jumps over the lazy dog",
        height=100,
        key="input_text",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        analyze_button = st.button("Analyze")
    with col2:
        auto_play = st.checkbox("Auto-play animations", value=True)

    if analyze_button:
        try:
            with st.spinner("Processing text..."):
                results = st.session_state["analyzer"].process_text(text)
                if results is None:
                    st.error("Text processing failed. Please check the input text and model configuration.")
                else:
                    st.session_state["results"] = results
                    st.session_state["analysis_complete"] = True

                    sequences = st.session_state["analyzer"].generate_text(results["input_ids"])
                    if sequences:
                        st.session_state["generated_sequences"] = sequences
                        st.session_state["generation_complete"] = True
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.session_state["analysis_complete"] = False
            logger.error(f"Analysis failed: {str(e)}")

    if st.session_state["analysis_complete"] and st.session_state["results"] is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Tokenization & Probabilities", "Attention Patterns", "Model Analysis", "Generated Output", "Advanced Analysis"]
        )

        with tab1:
            st.subheader("Tokenization Process")
            tokenization_fig = create_animated_tokenization(st.session_state["results"]["token_process"], auto_play=auto_play)
            st.plotly_chart(tokenization_fig, use_container_width=True)

            st.subheader("Token Distribution")
            token_df = pd.DataFrame(st.session_state["results"]["token_process"])
            st.bar_chart(token_df.groupby("token").size())

        with tab2:
            st.subheader("Attention Patterns")

            col1, col2 = st.columns(2)
            with col1:
                layer_idx = st.slider(
                    "Select Layer",
                    0,
                    st.session_state["analyzer"].config["num_layers"] - 1,
                    0,
                    key="layer_slider",
                )
            with col2:
                head_idx = st.slider(
                    "Select Head",
                    0,
                    st.session_state["analyzer"].config["num_heads"] - 1,
                    0,
                    key="head_slider",
                )

            attention_matrix = st.session_state["results"]["attentions"][layer_idx][0, head_idx].numpy()
            tokens = [
                st.session_state["analyzer"].tokenizer.decode(token_id)
                for token_id in st.session_state["results"]["input_ids"][0]
            ]

            fig = px.imshow(
                attention_matrix,
                labels=dict(x="Target Token", y="Source Token", color="Attention Weight"),
                x=tokens,
                y=tokens,
                color_continuous_scale="Viridis",
                title=f"Attention Pattern (Layer {layer_idx}, Head {head_idx})",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Model Statistics")
            metrics = evaluate_model(st.session_state["analyzer"], text)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Perplexity",
                    f"{metrics['perplexity']:.2f}" if not np.isnan(metrics["perplexity"]) else "N/A",
                )
            with col2:
                st.metric(
                    "Token Probability",
                    f"{metrics['avg_token_prob']:.2%}" if not np.isnan(metrics["avg_token_prob"]) else "N/A",
                )
            with col3:
                st.metric(
                    "Loss",
                    f"{metrics['loss']:.4f}" if not np.isnan(metrics["loss"]) else "N/A",
                )

        with tab4:
            if st.session_state.get("generation_complete", False):
                st.subheader("Generated Sequences")

                for i, sequence in enumerate(st.session_state["generated_sequences"]):
                    with st.expander(f"Generated Output {i+1}", expanded=(i == 0)):
                        st.write(sequence["text"])
                        st.write("---")
                        st.write("Token-by-token breakdown:")
                        st.json(sequence["tokens"])

        # ... (previous code remains unchanged until the Advanced Analysis tab)

        with tab5:
            st.subheader("Advanced Analysis")

            if st.session_state["results"]["attentions"] is not None:
                st.subheader("Feature Importance")
                feature_importance = st.session_state["analyzer"].calculate_feature_importance(st.session_state["results"]["attentions"])
                if feature_importance is not None:
                    # Ensure feature_importance is 1-dimensional
                    if feature_importance.ndim != 1:
                        st.error("Feature importance array is not 1-dimensional. Please check the data.")
                    else:
                        # Plot feature importance using Matplotlib
                        fig, ax = plt.subplots()
                        ax.bar(np.arange(len(feature_importance)), feature_importance)
                        ax.set_xlabel("Token Position")
                        ax.set_ylabel("Importance")
                        ax.set_title("Feature Importance by Token Position")
                        st.pyplot(fig)

            if st.session_state["results"]["hidden_states"] is not None:
                st.subheader("Behavioral Clustering")
                n_clusters = st.slider("Number of Clusters", 2, 5, 3)
                reduced_states, clusters = st.session_state["analyzer"].cluster_behavioral_patterns(st.session_state["results"]["hidden_states"], n_clusters=n_clusters)
                if reduced_states is not None:
                    # Ensure reduced_states is 2D and clusters is 1D
                    if reduced_states.ndim != 2 or clusters.ndim != 1:
                        st.error("Invalid data shape for clustering visualization.")
                    else:
                        # Create a DataFrame for visualization
                        cluster_df = pd.DataFrame({
                            "PCA Component 1": reduced_states[:, 0],
                            "PCA Component 2": reduced_states[:, 1],
                            "Cluster": clusters,
                        })

                        # Plot behavioral clustering
                        fig = px.scatter(
                            cluster_df,
                            x="PCA Component 1",
                            y="PCA Component 2",
                            color="Cluster",
                            labels={"x": "PCA Component 1", "y": "PCA Component 2"},
                            title="Behavioral Clustering of Hidden States",
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.subheader("Export Analysis")
            if st.button("Export Analysis Report"):
                report = {
                    "text": text,
                    "metrics": evaluate_model(st.session_state["analyzer"], text),
                    "token_process": st.session_state["results"]["token_process"],
                    "attentions": [att.numpy().tolist() for att in st.session_state["results"]["attentions"]],
                    "hidden_states": [hs.numpy().tolist() for hs in st.session_state["results"]["hidden_states"]],
                }
                report_json = json.dumps(report, indent=4)
                st.markdown(download_link(report_json, "analysis_report.json", "Download Analysis Report"), unsafe_allow_html=True)
else:
    st.warning("Please load a model first using the sidebar.")