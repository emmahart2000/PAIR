# Import Libraries
import statistics
import random
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras

def createMNISTPAIR(b_paired, x_paired, b_test, x_test, b_encoder, b_decoder, x_encoder, x_decoder):
    # Make Sure Inputs Are Correct Sizes
    if x_paired.shape[0] != b_paired.shape[0]:
        raise Exception('Must Input Paired Input and Target Images (same number of samples).')

    ## Note: sizes are hardcoded, matching the architecture of the network made in MNISTPAIRexample.py
    # 147 = 7*7*3 dimension of the latent space
    # 7 from original images size 28 divided by 2, divided by 2 (two convolutional layers)
    # 3 from number of channels)

    # Define Dimensions
    n_train = b_paired.shape[0]

    # Encode Paired Blurred Images
    latent_input_train = b_encoder.predict(b_paired)
    latent_input_train = np.transpose(latent_input_train.reshape((n_train, 147)))

    # Encode Paired Original Images
    latent_target_train = x_encoder.predict(x_paired)
    latent_target_train = np.transpose(latent_target_train.reshape((n_train, 147)))

    # Find Linear Mappings between Latent Variables from Paired Data
    inverse = latent_target_train @ linalg.pinv(latent_input_train)  # zx = mi zb
    forward = latent_input_train @ linalg.pinv(latent_target_train) # zb = mf zx
    np.save('variables/inverse', inverse)
    np.save('variables/forward', forward)

    # Invert Test Samples Through PAIR
    latent_input_test = b_encoder.predict(b_test)
    latent_input_test = np.transpose(latent_input_test.reshape((b_test.shape[0], 147)))
    latent_x_pred = inverse @ latent_input_test
    latent_x_pred = np.transpose(latent_x_pred)
    latent_x_pred = latent_x_pred.reshape(b_test.shape[0], 7, 7, 3)
    x_pred = x_decoder.predict(latent_x_pred)

    # Forward Propagate Test Samples Through PAIR
    latent_target_test = x_encoder.predict(x_test)
    latent_target_test = np.transpose(latent_target_test.reshape((b_test.shape[0], 147)))
    latent_b_pred = forward @ latent_target_test
    latent_b_pred = np.transpose(latent_b_pred)
    latent_b_pred = latent_b_pred.reshape(b_test.shape[0], 7, 7, 3)
    b_pred = b_decoder.predict(latent_b_pred)

    # Determine Error
    inv_errs = []
    for_errs = []
    for i in range(x_test.shape[0]):
        inv_err = x_test[i, :, :] - np.squeeze(x_pred[i, :, :])
        inv_errs.append(linalg.norm(inv_err) / linalg.norm(x_test[i, :, :]))
        for_err = b_test[i, :, :] - np.squeeze(b_pred[i, :, :])
        for_errs.append(linalg.norm(for_err) / linalg.norm(b_test[i, :, :]))
    inv_err = statistics.mean(inv_errs)
    for_err = statistics.mean(for_errs)

    # Plotting  Results------------------------------------------------------
    exs = 10
    js = random.sample(range(501), exs)
    fig, axs = plt.subplots(4, exs, figsize=(9, 4))

    # Create common color ranges for each row
    vmax_input = max((b_test[j].max() for j in js))
    vmin_input = min((b_test[j].min() for j in js))
    
    vmax_target = max((x_test[j].max() for j in js))
    vmin_target = min((x_test[j].min() for j in js))

    vmax_recon = max((x_pred[j].squeeze().max() for j in js))
    vmin_recon = min((x_pred[j].squeeze().min() for j in js))

    vmin_error = min(((abs(x_test[j] - x_pred[j].squeeze())).min() for j in js))
    vmax_error = max(((abs(x_test[j] - x_pred[j].squeeze())).max() for j in js))

    for i in range(exs):
        j = js[i]

        # Display Original Input Images (grayscale)
        ax = axs[0, i]
        im = ax.imshow(b_test[j], cmap='gray', vmin=vmin_input, vmax=vmax_input)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstructions (Autoencoder Outputs, grayscale)
        ax = axs[1, i]
        im = ax.imshow(x_pred[j].squeeze(), cmap='gray', vmin=vmin_recon, vmax=vmax_recon)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Target Images (grayscale)
        ax = axs[2, i]
        im = ax.imshow(x_test[j], cmap='gray', vmin=vmin_target, vmax=vmax_target)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstruction Errors (grayscale)
        ax = axs[3, i]
        im = ax.imshow(abs(x_test[j] - x_pred[j].squeeze()), cmap='gray', vmin=vmin_error, vmax=vmax_error)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Add colorbars to the right of each row
    for row in range(4):  # Now includes row 0 (Original Input Images)
        divider = make_axes_locatable(axs[row, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(axs[row, 0].images[0], cax=cax)  # Add colorbar for each row

    plt.suptitle('Example Inverted Images (error:' + str(round(inv_err, 4)) + ')')
    plt.savefig('images/inv_err.png')
    plt.show()

    fig, axs = plt.subplots(4, exs, figsize=(9, 4))

    vmax_recon = max((b_pred[j].squeeze().max() for j in js))
    vmin_recon = min((b_pred[j].squeeze().min() for j in js))

    vmin_error = min(((abs(b_test[j] - b_pred[j].squeeze())).min() for j in js))
    vmax_error = max(((abs(b_test[j] - b_pred[j].squeeze())).max() for j in js))

    for i in range(exs):
        j = js[i]

        # Display Original Input Images (grayscale)
        ax = axs[0, i]
        im = ax.imshow(x_test[j], cmap='gray', vmin=vmin_target, vmax=vmax_target)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstructions (Autoencoder Outputs, grayscale)
        ax = axs[1, i]
        im = ax.imshow(b_pred[j].squeeze(), cmap='gray', vmin=vmin_recon, vmax=vmax_recon)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Target Images (grayscale)
        ax = axs[2, i]
        im = ax.imshow(b_test[j], cmap='gray', vmin=vmin_input, vmax=vmax_input)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstruction Errors (grayscale)
        ax = axs[3, i]
        im = ax.imshow(abs(b_test[j] - b_pred[j].squeeze()), cmap='gray', vmin=vmin_error, vmax=vmax_error)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Add colorbars to the right of each row
    for row in range(4):  # Now includes row 0 (Original Input Images)
        divider = make_axes_locatable(axs[row, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(axs[row, 0].images[0], cax=cax)  # Add colorbar for each row

    plt.suptitle('Example Forward Propagated Images (error:' + str(round(for_err, 4)) + ')')
    plt.savefig('images/for_err.png')
    plt.show()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_pred[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('images/reconstructed_mnist_images.png')
    plt.show()

    return inverse, forward, inv_err, for_err
