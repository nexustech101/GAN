from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np

def build_gan(generator: models.Model, discriminator: models.Model, z_dimension: int) -> models.Model:
    """
    Builds the GAN model by combining the generator and discriminator.

    Args:
        generator (models.Model): The generator model.
        discriminator (models.Model): The discriminator model.
        z_dimension (int): Dimension of the noise vector.

    Returns:
        models.Model: The combined GAN model.
    """
    try:
        # Ensure the discriminator's weights are not updated during generator training
        discriminator.trainable = False

        # GAN input (noise vector)
        gan_input = layers.Input(shape=(z_dimension,))

        # Generate images
        generated_image = generator(gan_input)

        # Determine validity
        gan_output = discriminator(generated_image)

        # Combined GAN model
        gan = models.Model(gan_input, gan_output)

        return gan
    except (ValueError, TypeError) as e:
        print(f"Error building GAN: {e}")
        raise

# Example usage
if __name__ == "__main__":
    z_dim = 100  # Example z-dimension
    image_dim = 28 * 28  # Example image dimension

    generator = build_generator(z_dim, image_dim)
    discriminator = build_discriminator(image_dim)

    gan = build_gan(generator, discriminator, z_dim)
    gan.summary()
