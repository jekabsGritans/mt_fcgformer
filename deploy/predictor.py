import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.base_dataset import Transform
from models.base_model import BaseModel


class Predictor:
    def __init__(self, model: BaseModel, device: str, class_names: list[str] | None, transform: Transform):
        """
        Initialize the predictor.
        :param model: Model to use for prediction
        :param device: Device to use for prediction
        :param class_names: List of class names
        :param transform: Transform function to apply to the spectra to convert and prepare them for the model
        """
        self.transform = transform
        self.model = model.to(device)
        self.device = device

        if class_names is None:
            class_names = [f"Class {i}" for i in range(model.output_dim)]

        self.class_names = class_names

        self.model.eval()
        self.model.requires_grad_(False)

    def predict(self, spectrum: np.ndarray | torch.Tensor, threshold: float = 0.5, plot: bool = False) -> tuple[list[str], list[float]]:
        """
        Predict the functional groups for a given spectrum.
        :param spectrum: FTIR spectrum to predict. If a numpy array, given transform is applied first.
        :param threshold: Probability threshold for positive prediction
        :return: Tuple of (functional group names, probabilities) for positive predictions
        """

        assert spectrum.ndim == 1, f"Input spectrum must be 1d. Got {spectrum.ndim}d."
        if isinstance(spectrum, np.ndarray):
            spectrum = self.transform(spectrum)

        spectrum = spectrum.to(self.device)
        spectrum = spectrum.unsqueeze(0)  # Add batch dimension

        logits = self.model(spectrum)
        probs = torch.sigmoid(logits)
        probs = probs.squeeze(0)  # Remove batch dimension
        probs = probs.cpu().numpy()  # Move to CPU and convert to numpy array

        out_probs = []
        out_labels = []
        for i, prob in enumerate(probs):
            if prob >= threshold:
                out_probs.append(prob)
                out_labels.append(self.class_names[i])
        
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].plot(spectrum.cpu().numpy().flatten(), label="Spectrum")
            ax[0].set_title("FTIR Spectrum")
            ax[0].set_xlabel("Wavenumber")
            ax[0].set_ylabel("Absorbance")
            ax[0].legend()
            ax[1].bar(self.class_names, probs)

            ax[1].set_title("Predicted Probabilities")
            ax[1].set_xlabel("Functional Groups")
            ax[1].set_ylabel("Probability")
            ax[1].set_xticklabels(self.class_names, rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

        return out_labels, out_probs
