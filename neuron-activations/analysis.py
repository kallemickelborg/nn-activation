import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ActivationAnalysis:
    def __init__(self, activations, tokenizer, input_ids):
        """
        Initialize the ActivationAnalysis.

        Args:
            activations (dict): A dictionary of captured activations.
            tokenizer: The tokenizer used to process the input sentences.
            input_ids (torch.Tensor): The input IDs corresponding to the activations.
        """
        self.activations = activations
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.results = {}

    def compute_statistics(self):
        """
        Compute statistical measures on the activations.

        Returns:
            dict: A dictionary of statistical results.
        """
        statistics = {}
        for layer_name, activation in self.activations.items():
            activation_np = activation.cpu().numpy()
            mean = np.mean(
                activation_np, axis=(0, 1)
            )  # Mean over batch and sequence length
            std = np.std(activation_np, axis=(0, 1))
            statistics[layer_name] = {"mean": mean, "std": std}
        self.results["statistics"] = statistics
        return statistics

    def analyze_polysemanticity(self, threshold=0.5):
        """
        Analyze neurons for polysemanticity.

        Args:
            threshold (float): Activation threshold to consider a neuron as active.

        Returns:
            dict: Neurons identified as polysemantic in each layer.
        """
        polysemantic_neurons = {}
        for layer_name, activation in self.activations.items():
            activation_np = (
                activation.cpu().numpy()
            )  # Shape: (batch_size, seq_length, hidden_size)
            # Binarize activations per sample
            active = activation_np > threshold
            # Sum over samples to get activation counts per neuron per token
            active_counts = np.sum(active, axis=0)  # Shape: (seq_length, hidden_size)
            # Identify neurons that are active for multiple tokens
            num_tokens = active_counts.shape[0]
            neuron_activation_ratio = np.sum(active_counts > 0, axis=0) / num_tokens
            # Define polysemantic neurons as those active across multiple tokens
            poly_neurons = np.where(neuron_activation_ratio > 0.1)[0]
            polysemantic_neurons[layer_name] = poly_neurons
        self.results["polysemantic_neurons"] = polysemantic_neurons
        return polysemantic_neurons

    def get_token_activation_matrix(self, layer_name):
        """
        Get a matrix of activations per token for a specific layer.

        Args:
            layer_name (str): The layer to analyze.

        Returns:
            activation_matrix (np.ndarray): Shape (total_tokens, hidden_size)
            tokens (list): List of tokens corresponding to the activations.
        """
        activation = self.activations[layer_name]
        activation_np = activation.cpu().numpy()
        batch_size, seq_length, hidden_size = activation_np.shape
        # Flatten batch and sequence length
        activation_matrix = activation_np.reshape(batch_size * seq_length, hidden_size)
        # Get corresponding tokens
        input_ids_np = self.input_ids.cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids_np.flatten())
        return activation_matrix, tokens
