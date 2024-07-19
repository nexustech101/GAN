import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

def build_discriminator(image_dimension: int) -> models.Sequential:
    """
    Builds the discriminator model.

    Args:
        image_dimension (int): Dimension of the input image (flattened).

    Returns:
        models.Sequential: The discriminator model.
    """
    try:
        model = models.Sequential()

        # Input layer
        model.add(layers.Dense(128, input_dim=image_dimension))
        model.add(layers.LeakyReLU(alpha=0.2))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))

        return model
    except (ValueError, TypeError) as e:
        print(f"Error building discriminator: {e}")
        raise

# Example usage
if __name__ == "__main__":
    image_dim = 28 * 28
    discriminator = build_discriminator(image_dim)
    discriminator.summary()
