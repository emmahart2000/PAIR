# Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

def createMNISTexample(blur_kernel_size, sigma, noise_mean, noise_std, seed, figures):
    # blur_kernel_size = 8, sigma = 1.0, noise_mean = 0.0, noise_std = 0.1, seed=23, figures = 1
    # Load MNIST data
    (x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
    
    # Ensure the data has the right dimensions (batch_size, height, width, channels)
    x_train_val = x_train_val.astype(np.float32) / 255.0  # Normalize the data
    x_test = x_test.astype(np.float32) / 255.0
    x_train_val = np.expand_dims(x_train_val, axis=-1)  # Add channel dimension
    x_test = np.expand_dims(x_test, axis=-1)

    # Create Gaussian kernel for blurring
    def gaussian_kernel(size: int, sigma: float):
        """Creates a 2D Gaussian kernel array."""
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / np.sum(kernel)

    # Convert Gaussian kernel to a TensorFlow constant
    def get_gaussian_filter(size: int, sigma: float):
        kernel = gaussian_kernel(size, sigma)
        print(kernel)
        kernel = kernel[:, :, np.newaxis, np.newaxis]  # Shape (size, size, 1, 1)
        return tf.constant(kernel, dtype=tf.float32)

    # Apply Gaussian blur and add noise
    def process_image_tensorflow(images, blur_kernel_size, sigma, noise_mean, noise_std, seed):
        # Apply Gaussian blur using convolution
        gaussian_filter = get_gaussian_filter(blur_kernel_size, sigma)
        blurred_images = tf.nn.conv2d(images, gaussian_filter, strides=[1, 1, 1, 1], padding='SAME')

        # Add Gaussian noise
        tf.random.set_seed(seed)
        noise = tf.random.normal(shape=tf.shape(images), mean=noise_mean, stddev=noise_std)
        noisy_images = blurred_images + noise

        # # Clip the values to be within [0, 1]
        # noisy_images = tf.clip_by_value(noisy_images, 0.0, 1.0)
        # Normalize values to be within [0, 1]
        min_val = tf.reduce_min(noisy_images)
        max_val = tf.reduce_max(noisy_images)
        noisy_images = (noisy_images - min_val) / (max_val - min_val)

        return noisy_images

    # Convert NumPy arrays to TensorFlow tensors
    x_train_val_tf = tf.convert_to_tensor(x_train_val)
    x_test_tf = tf.convert_to_tensor(x_test)

    # Process the images
    b_train_val_tf = process_image_tensorflow(x_train_val_tf, blur_kernel_size, sigma, noise_mean, noise_std, seed)
    b_test_tf = process_image_tensorflow(x_test_tf, blur_kernel_size, sigma, noise_mean, noise_std, seed)

    # Convert the processed tensors back to NumPy arrays
    b_train_val = b_train_val_tf.numpy()
    b_test = b_test_tf.numpy()

    # Squeeze arrays to be compatible with later tasks
    x_train_val = np.squeeze(x_train_val, axis=-1)
    x_test = np.squeeze(x_test, axis=-1)
    b_train_val = np.squeeze(b_train_val, axis=-1)
    b_test = np.squeeze(b_test, axis=-1)

    # np.save('variables/x_train_val', x_train_val)
    # np.save('variables/x_test', x_train_val)
    # np.save('variables/b_train_val', x_train_val)
    # np.save('variables/b_test', x_train_val)

    if figures == 1:
        # Visualize some images in x_test
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(x_test[i], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('images/mnist_images.png')
        plt.show()

        # Visualize some images in b_test
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(b_test[i], cmap='gray')
            ax.axis('off')
        # plt.suptitle(f'blur_kernel_size = {blur_kernel_size}, sigma = {sigma}, noise_mean = {noise_mean}, noise_std = {noise_std}')
        plt.tight_layout()
        plt.savefig('images/blurred_mnist_images.png')
        plt.show()

    return x_train_val, x_test, b_train_val, b_test