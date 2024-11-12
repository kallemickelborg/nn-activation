from transformers import GPT2LMHeadModel
import torch.nn as nn


class ModelInterface:
    def __init__(self, model, framework="transformers"):
        """
        Initialize the ModelInterface.

        Args:
            model: The neural network model instance.
            framework (str): The framework ('pytorch', 'tensorflow', or 'transformers').
        """
        self.model = model
        self.framework = framework.lower()
        self.layers = self._extract_layers()

    def _extract_layers(self):
        """
        Extract layers from the model based on the framework.

        Returns:
            dict: A dictionary of layer names and layer objects.
        """
        layers = {}
        if self.framework in ["pytorch", "transformers"]:
            for name, module in self.model.named_modules():
                layers[name] = module
        elif self.framework == "tensorflow":
            # Implement TensorFlow layer extraction
            pass
        else:
            raise ValueError("Unsupported framework.")
        return layers

    def get_layer_by_name(self, name):
        """
        Retrieve a specific layer by name.

        Args:
            name (str): The name of the layer.

        Returns:
            The layer object.
        """
        return self.layers.get(name)
