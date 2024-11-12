from model_interface import ModelInterface
from activation_capture import ActivationCapture
from analysis import ActivationAnalysis


class AnalysisAgent:
    def __init__(self, model_interface, sentences, tokenizer, layers_to_monitor):
        """
        Initialize the AnalysisAgent.

        Args:
            model_interface (ModelInterface): The model interface.
            sentences (list): Input sentences for analysis.
            tokenizer: Tokenizer for the model.
            layers_to_monitor (list): Layers to capture activations from.
        """
        self.model_interface = model_interface
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.layers_to_monitor = layers_to_monitor
        self.activation_capture = ActivationCapture(model_interface)
        self.analysis = None
        self.state = "initialized"

    def create_plan(self):
        """
        Define the plan of actions.
        """
        self.plan = [
            self.capture_activations,
            self.perform_analysis,
            self.report_results,
        ]

    def execute_plan(self):
        """
        Execute the planned actions.
        """
        for action in self.plan:
            action()
            self.update_state(action.__name__)

    def update_state(self, action_name):
        """
        Update the agent's state based on the last action.
        """
        self.state = f"{action_name}_completed"
        print(f"State updated to: {self.state}")

    def capture_activations(self):
        """
        Capture activations from the model.
        """
        self.activation_capture.register_hooks(self.layers_to_monitor)
        inputs = self.tokenizer(
            self.sentences, return_tensors="pt", padding=True, truncation=True
        )
        self.activation_capture.capture(
            inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        self.activation_capture.remove_hooks()

    def perform_analysis(self):
        """
        Perform analysis on captured activations.
        """
        self.analysis = ActivationAnalysis(self.activation_capture.activations)
        self.analysis.compute_statistics()
        self.analysis.analyze_polysemanticity(threshold=0.5)

    def report_results(self):
        """
        Report the analysis results.
        """
        polysemantic_neurons = self.analysis.results["polysemantic_neurons"]
        for layer_name, neurons in polysemantic_neurons.items():
            print(f"Layer {layer_name} has {len(neurons)} polysemantic neurons.")
