# Utilities
from typing import List, Union, Tuple
import tensorflow as tf
import numpy as np
import logging
import models
import os

# Pyplot assets for model evaluation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output

# Model assets
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from models.generator.generator import build_generator
from models.discriminator.discriminator import build_discriminator
from models.gan.gan import build_gan

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameters
lr = 0.0001
beta1 = 0.3
beta2 = 0.9
z_dim = 100
image_dim = 28 * 28
batch_size = 300
epochs = 250
save_interval = 10

def load_and_preprocess_data() -> np.ndarray:
    """
    Loads and preprocesses the MNIST dataset.

    Returns:
        np.ndarray: Preprocessed training data.
    """
    try:
        (x_train, _), (_, _) = mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
        x_train = x_train.reshape(-1, image_dim)
        return x_train
    except (ValueError, IOError) as e:
        logging.error(f"Error loading MNIST dataset: {e}")
        raise

def compile_models(generator: models.Model, discriminator: models.Model) -> None:
    """
    Compiles the generator and discriminator models.

    Args:
        generator (models.Model): The generator model.
        discriminator (models.Model): The discriminator model.
    """
    try:
        discriminator.compile(optimizer=optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2),
                              loss='binary_crossentropy', metrics=['accuracy'])
        generator.compile(optimizer=optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2),
                          loss='binary_crossentropy')
    except (ValueError, TypeError) as e:
        logging.error(f"Error compiling models: {e}")
        raise

def train_discriminator(x_real: np.ndarray, y_real: np.ndarray, x_fake: np.ndarray, y_fake: np.ndarray) -> Tuple[float, float]:
    """
    Trains the discriminator on real and fake data.

    Args:
        x_real (np.ndarray): Real images.
        y_real (np.ndarray): Real labels.
        x_fake (np.ndarray): Fake images.
        y_fake (np.ndarray): Fake labels.

    Returns:
        Tuple[float, float]: Loss on real and fake data.
    """
    try:
        d_loss_real = discriminator.train_on_batch(x_real, y_real)
        d_loss_fake = discriminator.train_on_batch(x_fake, y_fake)
        return d_loss_real, d_loss_fake
    except tf.errors.InvalidArgumentError as e:
        logging.error(f"Error training discriminator: {e}")
        raise

def train_generator(generator: models.Model, gan: models.Model, batch_size: int) -> float:
    """
    Trains the generator via the GAN model.

    Args:
        generator (models.Model): The generator model.
        gan (models.Model): The combined GAN model.
        batch_size (int): Size of the training batch.

    Returns:
        float: Loss value of the generator.
    """
    try:
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)
        return g_loss
    except tf.errors.InvalidArgumentError as e:
        logging.error(f"Error training generator: {e}")
        raise

def save_generated_images(epoch: int, generator: models.Model, z_dim: int, save_dir: str = './images', num_images: int = 25) -> None:
    """
    Saves generated images to disk.

    Args:
        epoch (int): Current epoch number.
        generator (models.Model): The generator model.
        z_dim (int): Dimension of the noise vector.
        save_dir (str, optional): Directory to save images. Defaults to './images'.
        num_images (int, optional): Number of images to generate. Defaults to 25.
    """
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        noise = np.random.normal(0, 1, (num_images, z_dim))
        gen_imgs = generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images to [0, 1]

        fig, axs = plt.subplots(5, 5, figsize=(5, 5), sharey=True, sharex=True)
        cnt = 0
        
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[cnt].reshape(28, 28), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        
        fig.savefig(f"{save_dir}/gan_generated_image_epoch_{epoch}.png")
        plt.close()
    except (IOError, ValueError) as e:
        logging.error(f"Error saving generated images: {e}")
        raise

def build_gan_model(generator: models.Model, discriminator: models.Model, z_dim: int) -> models.Model:
    """
    Builds and compiles the GAN model.

    Args:
        generator (models.Model): The generator model.
        discriminator (models.Model): The discriminator model.
        z_dim (int): Dimension of the noise vector.

    Returns:
        models.Model: Compiled GAN model.
    """
    try:
        discriminator.trainable = False
        z = layers.Input(shape=(z_dim,))
        img = generator(z)
        validity = discriminator(img)
        gan = models.Model(z, validity)
        gan.compile(optimizer=optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2),
                    loss='binary_crossentropy')
        return gan
    except (ValueError, TypeError) as e:
        logging.error(f"Error building GAN model: {e}")
        raise

def save_models(generator: models.Model, discriminator: models.Model) -> None:
    """
    Saves the generator and discriminator models to disk.

    Args:
        generator (models.Model): The generator model.
        discriminator (models.Model): The discriminator model.
    """
    try:
        generator.save(r'./build/generator.keras')
        discriminator.save(r'./build/discriminator.keras')
        logging.info("Models saved successfully.")
    except (IOError, ValueError) as e:
        logging.error(f"Error saving models: {e}")
        raise

# Main training script
if __name__ == "__main__":
    logging.info("Loading and preprocessing data...")
    x_train = load_and_preprocess_data()

    logging.info("Building models...")
    try:
        discriminator = build_discriminator(image_dim)
        generator = build_generator(z_dim, image_dim)
    except (ValueError, TypeError) as e:
        logging.error(f"Error building models: {e}")
        raise

    logging.info("Compiling models...")
    compile_models(generator, discriminator)

    logging.info("Building GAN model...")
    gan = build_gan_model(generator, discriminator, z_dim)

    logging.info("Starting training loop...")
    try:
        for epoch in range(epochs):
            for i in range(x_train.shape[0] // batch_size):
                # Train Discriminator
                noise = tf.random.normal(shape=(batch_size, z_dim), mean=0.0, stddev=1.0)
                gen_imgs = generator.predict(noise)
                
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_imgs = x_train[idx]

                d_loss_real, d_loss_fake = train_discriminator(real_imgs, np.ones((batch_size, 1)), gen_imgs, np.zeros((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train Generator
                g_loss = train_generator(generator, gan, batch_size)

                logging.info(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]}] [G loss: {g_loss}]")

            if epoch % save_interval == 0:
                save_generated_images(epoch, generator, z_dim)

    except (KeyboardInterrupt, Exception) as e:
        logging.error(f"Training interrupted: {e}")
        save_models(generator, discriminator)
    
    save_models(generator, discriminator)