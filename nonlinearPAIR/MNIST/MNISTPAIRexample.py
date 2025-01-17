# Import Libraries
import numpy as np
from tensorflow import keras
from createMNISTexample import createMNISTexample
from createMNISTautoencoder import createMNISTautoencoder
from createMNISTPAIR import createMNISTPAIR
from createMNISTAEFigures import createMNISTAEFigures
from notMNIST import notMNIST
from OODmetrics import OODmetrics

# Create MNIST Deblurring Example--------------------------------------------------------------------------------------
x_train_val, x_test, b_train_val, b_test = createMNISTexample(blur_kernel_size = 8,
                                                              sigma = 10, 
                                                              noise_mean = 0.0, 
                                                              noise_std = 0.01, 
                                                              seed=23, 
                                                              figures = 1)


# # # Create Autoencoders (Unsupervised Tasks)-----------------------------------------------------------------------------
# lr_sched = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries = [50, 100, 150, 200, 250, 300, 350], 
#                                                              values = [1e-3, 1e-4, 1e-3, 1e-4, 1e-3, 1e-4, 1e-3, 1e-4])
# lr_sched = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries = [100, 200, 300], 
                                                            #  values = [1e-3, 1e-4, 1e-3, 1e-4])
# # lr_sched = keras.optimizers.schedules.PiecewiseConstantDecay(boundaries = [100, 200, 300], 
# #                                                              values = [1e-2, 1e-3, 1e-4, 1e-5])
# # lr_sched = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
# #                                                            decay_steps=100,
# #                                                            decay_rate=0.9)
# createMNISTautoencoder(b_train_val, ae_optimizer = keras.optimizers.Adam(learning_rate=lr_sched), epoch_num=400, label='b')
# createMNISTautoencoder(x_train_val, ae_optimizer = keras.optimizers.Adam(learning_rate=lr_sched), epoch_num=400, label='x')
b_encoder = keras.models.load_model('models/MNIST_b_encoder.keras')
b_decoder = keras.models.load_model('models/MNIST_b_decoder.keras')
x_encoder = keras.models.load_model('models/MNIST_x_encoder.keras')
x_decoder = keras.models.load_model('models/MNIST_x_decoder.keras')

# xae_err, bae_err = createMNISTAEFigures(x_test, b_test, b_encoder, b_decoder, x_encoder, x_decoder)

# # Create PAIR (Supervised Task)----------------------------------------------------------------------------------------
# # Determine PAIR Errors
inverse, forward, inv_error, for_error = createMNISTPAIR(b_train_val, x_train_val, b_test, x_test, b_encoder, b_decoder, x_encoder, x_decoder)

# # Determine OOD Metrics
x_out, b_out = notMNIST(blur_kernel_size = 8, sigma = 10, noise_mean = 0.0, noise_std = 0.01, seed=23)

OODmetrics(b_test[:5000,:,:], b_out, x_test[:5000,:,:], x_out, b_encoder, b_decoder, x_encoder, x_decoder, inverse, forward)

