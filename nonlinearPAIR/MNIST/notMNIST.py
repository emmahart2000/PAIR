import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def notMNIST(blur_kernel_size = 5, sigma = 1.0, noise_mean = 0.0, noise_std = 0.5, seed=23):
    x_out = np.zeros((5000, 28, 28))
    folders = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    current_index = 0
    for folder in folders:
        folder_path = os.path.join("/local/scratch/ehart5/PAIR/nonlinearPAIR/MNIST/notMNIST", folder)
        image_files = os.listdir(folder_path)

        for i, img_file in enumerate(image_files[:500]):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            x_out[current_index, :, :] = img
            current_index = current_index + 1

    # Check the shape of the output array
    print("Shape of x_out:", x_out.shape)
    x_out = np.float32(x_out)
    print(type(x_out[1,1,1]))
    print(type(x_out))

    x_out = np.expand_dims(x_out, axis=-1)

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
    x_out_tf = tf.convert_to_tensor(x_out)

    # Process the images
    b_out_tf = process_image_tensorflow(x_out_tf, blur_kernel_size, sigma, noise_mean, noise_std, seed)

    # Convert the processed tensors back to NumPy arrays
    b_out = b_out_tf.numpy()

    # Squeeze arrays to be compatible with later tasks
    x_out = np.squeeze(x_out, axis=-1)
    b_out = np.squeeze(b_out, axis=-1)

    # Visualize some images in x_test
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_out[i*500+1], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('images/notmnist_images.png')
    plt.show()

    # Visualize some images in b_test
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(b_out[i*500+1], cmap='gray')
        ax.axis('off')
    # plt.suptitle(f'blur_kernel_size = {blur_kernel_size}, sigma = {sigma}, noise_mean = {noise_mean}, noise_std = {noise_std}')
    plt.tight_layout()
    plt.savefig('images/blurred_notmnist_images.png')
    plt.show()
    return x_out, b_out
