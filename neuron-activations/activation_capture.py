import torch


class ActivationCapture:
    def __init__(self, model_interface):
        """
        Initialize the ActivationCapture.

        Args:
            model_interface (ModelInterface): An instance of ModelInterface.
        """
        self.model_interface = model_interface
        self.activations = {}
        self.handles = []

    def _get_activation_hook(self, layer_name):
        def hook(module, input, output):
            # Store activations with gradient tracking disabled
            self.activations[layer_name] = output.detach()

        return hook

    def register_hooks(self, layers):
        """
        Register hooks or callbacks to capture activations.

        Args:
            layers (list): A list of layer names to monitor.
        """
        for layer_name in layers:
            layer = self.model_interface.get_layer_by_name(layer_name)
            if layer is not None:
                handle = layer.register_forward_hook(
                    self._get_activation_hook(layer_name)
                )
                self.handles.append(handle)
            else:
                print(f"Layer {layer_name} not found in the model.")

    def remove_hooks(self):
        """
        Remove all registered hooks.
        """
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def capture(self, inputs, attention_mask=None, device="cpu"):
        """
        Run the model and capture activations.

        Args:
            inputs: The input IDs for the model.
            attention_mask: The attention mask.
            device: The device to run the model on ('cpu' or 'cuda').

        Returns:
            The model's output logits.
        """
        self.model_interface.model.to(device)
        self.model_interface.model.eval()
        with torch.no_grad():
            inputs = inputs.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                outputs = self.model_interface.model(
                    inputs, attention_mask=attention_mask
                )
            else:
                outputs = self.model_interface.model(inputs)
        return outputs.logits
