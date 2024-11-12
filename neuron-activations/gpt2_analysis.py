from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

from model_interface import ModelInterface
from activation_capture import ActivationCapture
from analysis import ActivationAnalysis
from visualization import ActivationVisualizer
from agentic_ai import AnalysisAgent


def main():
    # Load GPT-2 small model and tokenizer
    model_name = "gpt2"  # GPT-2 small
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    # Initialize the Model Interface
    model_interface = ModelInterface(model, framework="transformers")

    # Specify layers to monitor (e.g., the feed-forward layers in each transformer block)
    layers_to_monitor = [
        f"transformer.h.{i}.mlp.c_fc" for i in range(model.config.n_layer)
    ]

    # Initialize the Activation Capture
    activation_capture = ActivationCapture(model_interface)
    activation_capture.register_hooks(layers_to_monitor)

    # Prepare input data (a batch of sentences)
    sentences = [
        "The cat sat on the mat.",
        "Artificial intelligence is transforming industries.",
        "The stock market fluctuates unpredictably.",
        "Climate change is a pressing global issue.",
        "Medieval knights wore armor in battle.",
        "Computers use binary code to process information.",
        "Music has the power to evoke emotions.",
        "The universe is vast and largely unexplored.",
        "Cooking requires precise measurements for baking.",
    ]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]

    # Capture activations
    outputs = activation_capture.capture(
        input_ids, attention_mask=inputs["attention_mask"]
    )

    # Remove hooks after capturing
    activation_capture.remove_hooks()

    # Analyze activations
    analysis = ActivationAnalysis(
        activations=activation_capture.activations,
        tokenizer=tokenizer,
        input_ids=input_ids,
    )
    statistics = analysis.compute_statistics()
    polysemantic_neurons = analysis.analyze_polysemanticity(threshold=0.5)

    # Visualize activations
    visualizer = ActivationVisualizer(analysis)

    # Select a layer to visualize
    layer_to_visualize = "transformer.h.5.mlp.c_fc"

    # Plot heatmap of neuron activations across tokens
    visualizer.plot_neuron_activation_heatmap(layer_to_visualize)

    # Highlight polysemantic neurons
    visualizer.highlight_polysemantic_neurons(layer_to_visualize)

    # Plot neuron activations for a specific token
    # Let's pick the first token
    visualizer.plot_token_neuron_activation(layer_to_visualize, token_index=0)


if __name__ == "__main__":
    main()
