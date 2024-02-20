# Import Libraries
import statistics
import skimage as ski
import numpy as np
from numpy import linalg
from tensorflow import keras

def createNonlinearPAIR(paired_input, paired_target, input_test, target_test):
    # Make Sure Inputs Are Correct Sizes
    if paired_target.shape[0] != paired_input.shape[0]:
        raise Exception('Must Input Paired Input and Target Images (same number of samples).')

    ## Note: sizes are hardcoded, matching the architecture of the network made in nonlinearPAIRexample.py
    # 147 = 7*7*3 dimension of the latent space
    # 7 from original images size 28 divided by 2, divided by 2 (two convolutional layers)
    # 3 from number of channels)

    # Define Dimensions
    n_train = paired_input.shape[0]

    # Load Previously Made Autoencoder Networks, Trained on all Unpaired Training Samples
    b_encoder = keras.models.load_model('models/MNIST_input_encoder')
    b_decoder = keras.models.load_model('models/MNIST_input_decoder')
    x_encoder = keras.models.load_model('models/MNIST_target_encoder')
    x_decoder = keras.models.load_model('models/MNIST_target_decoder')

    # Encode Paired Blurred Images
    latent_input_train = b_encoder.predict(paired_input)
    latent_input_train = np.transpose(latent_input_train.reshape((n_train, 147)))

    # Encode Paired Original Images
    latent_target_train = x_encoder.predict(paired_target)
    latent_target_train = np.transpose(latent_target_train.reshape((n_train, 147)))

    # Find Linear Mappings between Latent Variables from Paired Data
    inverse = latent_target_train @ linalg.pinv(latent_input_train)  # zx = mi zb
    forward = latent_input_train @ linalg.pinv(latent_target_train) # zb = mf zx

    # Invert Test Samples Through PAIR
    latent_input_test = b_encoder.predict(b_test)
    latent_input_test = np.transpose(latent_input_test.reshape((b_test.shape[0], 147)))
    latent_x_pred = inverse @ latent_input_test
    latent_x_pred = np.transpose(latent_x_pred)
    latent_x_pred = latent_x_pred.reshape(b_test.shap[0], 7, 7, 3)
    x_pred = decoder.predict(latent_x_pred)

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
        for_errs.append(linalg.norm(for_err) / linalg.norm(b_test[i, :, ;]))
    inv_err = statistics.mean(errs)
    for_err = statistics.mean(errs)

    return inv_err, for_err
