# Import Packages
from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import numpy as np
from numpy import linalg
import statistics

def createMNISTAEFigures(x_test, b_test, b_encoder, b_decoder, x_encoder, x_decoder):
    x_pred = x_decoder.predict(x_encoder.predict(x_test))
    b_pred = b_decoder.predict(b_encoder.predict(b_test))

    # Determine Autoencoder Errors
    x_errs = []
    b_errs = []
    for i in range(x_test.shape[0]):
        x_err = x_test[i, :, :] - np.squeeze(x_pred[i, :, :])
        x_errs.append(linalg.norm(x_err) / linalg.norm(x_test[i, :, :]))
        b_err = b_test[i, :, :] - np.squeeze(b_pred[i, :, :])
        b_errs.append(linalg.norm(b_err) / linalg.norm(b_test[i, :, :]))
    xae_err = statistics.mean(x_errs)
    bae_err = statistics.mean(b_errs)

    # Plotting  Results------------------------------------------------------
    exs = 5
    js = random.sample(range(501), 5)
    fig, axs = plt.subplots(3, exs, figsize=(9, 4), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # Create common color ranges for each row
    vmax_input = max((x_test[j].max() for j in js))
    vmin_input = min((x_test[j].min() for j in js))

    vmax_recon = max((x_pred[j].squeeze().max() for j in js))
    vmin_recon = min((x_pred[j].squeeze().min() for j in js))

    vmin_error = min(((abs(x_test[j] - x_pred[j].squeeze())).min() for j in js))
    vmax_error = max(((abs(x_test[j] - x_pred[j].squeeze())).max() for j in js))

    for i in range(exs):
        j = js[i]

        # Display Original Input Images (grayscale)
        ax = axs[0, i]
        im = ax.imshow(x_test[j], cmap='gray', vmin=vmin_input, vmax=vmax_input)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstructions (Autoencoder Outputs, grayscale)
        ax = axs[1, i]
        im = ax.imshow(x_pred[j].squeeze(), cmap='gray', vmin=vmin_recon, vmax=vmax_recon)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstruction Errors (grayscale)
        ax = axs[2, i]
        im = ax.imshow(abs(x_test[j] - x_pred[j].squeeze()), cmap='gray', vmin=vmin_error, vmax=vmax_error)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Add colorbars to the right of each row
    for row in range(3):  # Now includes row 0 (Original Input Images)
        divider = make_axes_locatable(axs[row, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(axs[row, 0].images[0], cax=cax)  # Add colorbar for each row

    plt.suptitle('Example Autoencoded Target Images (error:' + str(round(xae_err, 4)) + ')')
    plt.savefig('images/xae_err.png')
    plt.show()

    fig, axs = plt.subplots(3, exs, figsize=(9, 4), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # Create common color ranges for each row
    vmin_input = min((b_test[j].min() for j in range(500)))
    vmax_input = max((b_test[j].max() for j in range(500)))

    # vmin_recon = min((b_pred[j].squeeze().min() for j in range(500)))
    # vmax_recon = max((b_pred[j].squeeze().max() for j in range(500)))
    vmin_recon = 0
    vmax_recon = 1

    # vmin_error = min(((abs(b_test[j] - b_pred[j].squeeze())).min() for j in range(500)))
    # vmax_error = max(((abs(b_test[j] - b_pred[j].squeeze())).max() for j in range(500)))
    vmin_error = 0
    vmax_error = 1
    
    for i in range(exs):
        j = js[i]

        # Display Original Input Images (grayscale)
        ax = axs[0, i]
        im = ax.imshow(b_test[j], cmap='gray', vmin=vmin_input, vmax=vmax_input)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstructions (Autoencoder Outputs, grayscale)
        ax = axs[1, i]
        im = ax.imshow(b_pred[j].squeeze(), cmap='gray', vmin=vmin_recon, vmax=vmax_recon)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display Reconstruction Errors (grayscale)
        ax = axs[2, i]
        im = ax.imshow(abs(b_test[j] - b_pred[j].squeeze()), cmap='gray', vmin=vmin_error, vmax=vmax_error)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Add colorbars to the right of each row
    for row in range(3):  # Now includes row 0 (Original Input Images)
        divider = make_axes_locatable(axs[row, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(axs[row, 0].images[0], cax=cax)  # Add colorbar for each row

    plt.suptitle('Example Autoencoded Input Images (error:' + str(round(xae_err, 4)) + ')')
    plt.savefig('images/bae_err.png')
    plt.show()

    return xae_err, bae_err

