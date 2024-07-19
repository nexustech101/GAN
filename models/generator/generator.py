import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

def build_generator(z_dimension: int, image_dimension: int) -> models.Sequential:
    """
    Builds the generator model.

    Args:
        z_dimension (int): Dimension of the noise vector.
        image_dimension (int): Dimension of the generated image (flattened).

    Returns:
        models.Sequential: The generator model.
    """
    try:
        model = Sequential()

        # Input layer
        model.add(layers.Dense(128, input_dim=z_dimension))
        model.add(layers.LeakyReLU(alpha=0.2))

        # Output layer
        model.add(layers.Dense(image_dimension, activation='tanh'))

        return model
    except (ValueError, TypeError) as e:
        print(f"Error building generator: {e}")
        raise

# Example usage
if __name__ == "__main__":
    z_dim = 100
    image_dim = 28 * 28
    generator = build_generator(z_dim, image_dim)
    generator.summary()
