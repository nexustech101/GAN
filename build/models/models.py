import tensorflow as tf
from tensorflow.keras import models
from typing import Optional

class Generator:
    """
    A class representing a pre-trained generator model for a Generative Adversarial Network (GAN).
    
    Attributes:
        model (tf.keras.Model): The pre-trained generator model.
    """

    def __init__(self, model_path: str):
        """
        Initializes the Generator class by loading a pre-trained model from the specified path.
        Args:
            model_path (str): The path to the saved generator model.
        Raises:
            ValueError: If the model file cannot be loaded.
        """
        try:
            self.model = models.load_model(model_path)
        except (OSError, ValueError) as e:
            raise ValueError(f"Failed to load the generator model from {model_path}: {e}")

    def generate(self, batch_size: int) -> tf.Tensor:
        """
        Generates a batch of images using the pre-trained generator model.
        Args:
            batch_size (int): The number of images to generate.
        Returns:
            tf.Tensor: A tensor containing the generated images.
        Raises:
            ValueError: If the batch size is not a positive integer.
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        
        noise_dim = self.model.input_shape[1]
        noise = tf.random.normal([batch_size, noise_dim])
        return self.model(noise)


class Discriminator:
    """
    A class representing a pre-trained discriminator model for a Generative Adversarial Network (GAN).

    Attributes:
        model (tf.keras.Model): The pre-trained discriminator model.
    """

    def __init__(self, model_path: str):
        """
        Initializes the Discriminator class by loading a pre-trained model from the specified path.
        Args:
            model_path (str): The path to the saved discriminator model.
        Raises:
            ValueError: If the model file cannot be loaded.
        """
        try:
            self.model = models.load_model(model_path)
        except (OSError, ValueError) as e:
            raise ValueError(f"Failed to load the discriminator model from {model_path}: {e}")
