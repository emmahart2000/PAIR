# Import Libraries
import statistics
from PIL import Image
import numpy as np
from numpy import linalg
from tensorflow import keras
from keras import layers
from keras import backend as K
from createHTPAIR import createHTPAIR

# Load HT Input Images
n_imgs = 10000
b_all = np.zeros((n_imgs, 100, 100), dtype=np.uint8)
x_all = np.zeros((n_imgs, 20, 19), dtype=np.uint8)

for i in range(1,n_imgs+1):
    b_filename = f"images/input{i}.png"
    x_filename = f"images/target{i}.png"
    b_img = Image.open(b_filename)
    x_img = Image.open(x_filename)
    b_all[i-1] = np.array(b_img) 
    x_all[i-1] = np.array(x_img)

b_train_val = b_all[0:9000,:,:]
x_train_val = x_all[0:9000,:,:]
b_test = b_all[9000:10001,:,:]
x_test = x_all[9000:10001,:,:]

# Create PAIR Autoencoders (Unsupervised Task)
# Blurred Autoencoder--------------------------------------------------------------------------------------------------
n_train = b_train_val.shape[0]
split = round((5/6)*n_train)

# Split Paired Data into Training and Validation
b_train = b_train_val[:split, :, :]
b_val = b_train_val[split:, :, :]

# Establish Architecture of the Encoder
latent_dim = 3
encoder_input = keras.Input(shape=(28, 28, 1), name='original_image')
inner_encoded = layers.Conv2D(np.ceil(latent_dim / 2), 3, activation='relu', padding='same')(encoder_input)
inner_encoded = layers.MaxPooling2D(2, padding='same')(inner_encoded)
inner_encoded = layers.Conv2D(latent_dim, 3, activation='relu', padding='same')(inner_encoded)
encoder_output = layers.MaxPooling2D(2, padding='same')(inner_encoded)
b_encoder = keras.Model(encoder_input, encoder_output, name='b_encoder')
b_encoder.summary()

# Establish Architecture of the Decoder
decoder_input = keras.Input(shape=(7, 7, latent_dim), name='encoded_image')
inner_decoded = layers.Conv2DTranspose(latent_dim, 3, activation='relu', padding='same')(decoder_input)
inner_decoded = layers.UpSampling2D(2)(inner_decoded)
inner_decoded = layers.Conv2DTranspose(np.ceil(latent_dim / 2), 3, activation='relu', padding='same')(inner_decoded)
inner_decoded = layers.UpSampling2D(2)(inner_decoded)
decoder_output = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(inner_decoded)
b_decoder = keras.Model(decoder_input, decoder_output, name='b_decoder')
b_decoder.summary()

# Create Full Autoencoder Model
autoencoder_input = keras.Input(shape=(28, 28, 1), name='original_image')
encoded_image = b_encoder(autoencoder_input)
decoded_image = b_decoder(encoded_image)
b_autoencoder = keras.Model(autoencoder_input, decoded_image, name='b_autoencoder')
b_autoencoder.summary()

# Compile Autoencoder
b_autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                         loss='mse')
b_autoencoder.fit(b_train, b_train,
                epochs=25,
                batch_size=256,
                validation_data=(b_val, b_val))
K.set_value(b_autoencoder.optimizer.learning_rate, 1e-4)
b_autoencoder.fit(b_train, b_train,
                epochs=25,
                batch_size=256,
                validation_data=(b_val, b_val))

# Save Model
b_autoencoder.save('models/MNIST_input_ae')
b_encoder.save('models/MNIST_input_encoder')
b_decoder.save('models/MNIST_input_decoder')

# Original Autoencoder ------------------------------------------------------------------------------------------------
x_train = x_train_val[:split, :, :]
x_val = x_train_val[split:, :, :]

# Establish Architecture of the Encoder
latent_dim = 3
encoder_input = keras.Input(shape=(28, 28, 1), name='original_image')
inner_encoded = layers.Conv2D(np.ceil(latent_dim / 2), 3, activation='relu', padding='same')(encoder_input)
inner_encoded = layers.MaxPooling2D(2, padding='same')(inner_encoded)
inner_encoded = layers.Conv2D(latent_dim, 3, activation='relu', padding='same')(inner_encoded)
encoder_output = layers.MaxPooling2D(2, padding='same')(inner_encoded)
x_encoder = keras.Model(encoder_input, encoder_output, name='x_encoder')
x_encoder.summary()

# Establish Architecture of the Decoder
decoder_input = keras.Input(shape=(7, 7, latent_dim), name='encoded_image')
inner_decoded = layers.Conv2DTranspose(latent_dim, 3, activation='relu', padding='same')(decoder_input)
inner_decoded = layers.UpSampling2D(2)(inner_decoded)
inner_decoded = layers.Conv2DTranspose(np.ceil(latent_dim / 2), 3, activation='relu', padding='same')(inner_decoded)
inner_decoded = layers.UpSampling2D(2)(inner_decoded)
decoder_output = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(inner_decoded)
x_decoder = keras.Model(decoder_input, decoder_output, name='x_decoder')
x_decoder.summary()

# Create Full Autoencoder Model
autoencoder_input = keras.Input(shape=(28, 28, 1), name='original_image')
encoded_image = x_encoder(autoencoder_input)
decoded_image = x_decoder(encoded_image)
x_autoencoder = keras.Model(autoencoder_input, decoded_image, name='x_autoencoder')
x_autoencoder.summary()

# Compile Autoencoder
x_autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                         loss='mse')
x_autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                validation_data=(x_val, x_val))
K.set_value(x_autoencoder.optimizer.learning_rate, 1e-4)
x_autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                validation_data=(x_val, x_val))

# Save
x_autoencoder.save('models/MNIST_target_ae')
x_encoder.save('models/MNIST_target_encoder')
x_decoder.save('models/MNIST_target_decoder')

# Autoencode Test Images
x_pred = x_autoencoder.predict(x_test)
b_pred = b_autoencoder.predict(b_test)

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

print('x autoencoder average relative test error')
print(xae_err)

print('b autoencoder average relative test error')
print(bae_err)


# Determine PAIR Errors
inv_error, for_error = createNonlinearPAIR(b_train_val, x_train_val, b_test, x_test)
print('PAIR inv error')
print(inv_error)

print('PAIR for error')
print(for_error)
