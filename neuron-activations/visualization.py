import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ActivationVisualizer:
    def __init__(self, analysis_results):
        """
        Initialize the ActivationVisualizer.

        Args:
            analysis_results (ActivationAnalysis): An instance of ActivationAnalysis.
        """
        self.analysis_results = analysis_results

    def plot_neuron_activation_heatmap(
        self, layer_name, neuron_indices=None, max_neurons=50
    ):
        """
        Plot a heatmap of neuron activations across tokens.

        Args:
            layer_name (str): The layer to visualize.
            neuron_indices (list or np.ndarray): Indices of neurons to include.
            max_neurons (int): Maximum number of neurons to display.
        """
        activation_matrix, tokens = self.analysis_results.get_token_activation_matrix(
            layer_name
        )

        # Clean up token labels
        cleaned_tokens = [
            token.replace("Ġ", "").replace("<|endoftext|>", "EOṠ") for token in tokens
        ]

        if neuron_indices is None:
            # Select top neurons based on variance
            variances = np.var(activation_matrix, axis=0)
            neuron_indices = np.argsort(variances)[-max_neurons:]
        else:
            neuron_indices = np.array(neuron_indices)

        # Subset activation matrix
        activation_subset = activation_matrix[:, neuron_indices]

        # Normalize activations to [0,1] range
        activation_subset = (activation_subset - activation_subset.min()) / (
            activation_subset.max() - activation_subset.min()
        )

        # Create heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            activation_subset.T,
            xticklabels=cleaned_tokens,
            yticklabels=neuron_indices,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        plt.title(f"Neuron Activations for Layer {layer_name}")
        plt.xlabel("Tokens")
        plt.ylabel("Neuron Index")
        plt.xticks(rotation=90)
        plt.show()

    def highlight_polysemantic_neurons(self, layer_name):
        """
        Plot activations highlighting polysemantic neurons.

        Args:
            layer_name (str): The layer to visualize.
        """
        polysemantic_neurons = self.analysis_results.results.get(
            "polysemantic_neurons", {}
        )
        neuron_indices = polysemantic_neurons.get(layer_name, None)
        if neuron_indices is not None and len(neuron_indices) > 0:
            self.plot_neuron_activation_heatmap(
                layer_name, neuron_indices=neuron_indices
            )
        else:
            print(f"No polysemantic neurons found in layer {layer_name}.")

    def plot_token_neuron_activation(self, layer_name, token_index):
        """
        Plot neuron activations for a specific token.

        Args:
            layer_name (str): The layer to visualize.
            token_index (int): The index of the token in the input sequence.
        """
        activation_matrix, tokens = self.analysis_results.get_token_activation_matrix(
            layer_name
        )
        if token_index >= len(tokens):
            print("Token index out of range.")
            return
        activation_values = activation_matrix[token_index]
        neuron_indices = np.arange(len(activation_values))
        plt.figure(figsize=(12, 6))
        plt.bar(neuron_indices, activation_values)
        plt.title(
            f'Neuron Activations for Token "{tokens[token_index]}" in Layer {layer_name}'
        )
        plt.xlabel("Neuron Index")
        plt.ylabel("Activation Value")
        plt.show()
