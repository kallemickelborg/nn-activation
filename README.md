# Neural Network - Neuron Activation Analysis

(WIP!) Project on neuron activations in neural networks using Agentic AI concepts inspired by the AutoGen and StateFlow papers. This project is meant to be a tool for analyzing and visualizing neuron activations in neural networks, with a focus on understanding polysemantic neurons and activation patterns. In the project, I have used the GPT-2-small model from Hugging Face. You can easily swap it out for any other model.

## Features

- Model Interface for PyTorch and TensorFlow

  - Unified interface for different deep learning frameworks
  - Layer access and activation monitoring
  - Support for transformer-based models

- Activation Capture Mechanisms

  - Non-intrusive hook-based activation capture
  - Real-time monitoring of neural network layers
  - Memory-efficient activation storage

- Analysis Tools for Activations and Logits

  - Statistical analysis of neuron behaviors
  - Polysemantic neuron detection
  - Token-level activation analysis
  - Clustering and pattern recognition

- Agentic AI Module for Automated Analysis

  - State-based analysis workflow
  - Automated insight generation
  - Configurable analysis pipelines

- Visualization Tools for Insights
  - Activation heatmaps
  - Polysemantic neuron highlighting
  - Token-specific activation plots
  - Interactive visualization options

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run the project, you can use the `gpt2_analysis.py` file. As it currently is, it will load the GPT-2-small model from Hugging Face, capture activations from the feed-forward layers in each transformer block based on the few sentences in the `sentences` list, and analyze the activations for polysemantic neurons.

Below is a minimal example of how you can set up the project to analyze activations from a different model.

```python
from neuron_activations import ModelInterface, ActivationMonitor

# Initialize with a PyTorch model
model = YourModel()
interface = ModelInterface(model, framework='pytorch')

# Set up activation monitoring
monitor = ActivationMonitor(interface)
monitor.register_layers(['layer1', 'layer2.attention'])

# Run inference and capture activations
input_text = "Example input for analysis"
activations = monitor.capture(input_text)

# Analyze activations
from neuron_activations.analysis import PolysematicDetector

detector = PolysematicDetector(activations)
polysemantic_neurons = detector.detect()

# Visualize results
from neuron_activations.viz import ActivationVisualizer

viz = ActivationVisualizer(activations)
viz.plot_heatmap(layer='layer1')
viz.highlight_polysemantic(polysemantic_neurons)
```

## Contributing

WIP! Contributing details will come later.

## License

[MIT License](LICENSE)
