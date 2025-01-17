# Import Libraries
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

def createMNISTautoencoder(y_train_val, ae_optimizer = keras.optimizers.Adam(learning_rate=1e-3), epoch_num=200, label='XXX'):
    n_train = y_train_val.shape[0]
    split = round((5/6)*n_train)

    # Split Paired Data into Training and Validation
    y_train = y_train_val[:split, :, :]
    y_val = y_train_val[split:, :, :]

    # Establish Architecture of the Encoder
    channel_max = 3
    encoder_input = keras.Input(shape=(28, 28, 1), name='original_image')
    inner_encoded = layers.Conv2D(int(channel_max / 2), 3, activation='relu', padding='same')(encoder_input)
    inner_encoded = layers.MaxPooling2D(2, padding='same')(inner_encoded)
    inner_encoded = layers.Conv2D(channel_max, 3, activation='relu', padding='same')(inner_encoded)
    encoder_output = layers.MaxPooling2D(2, padding='same')(inner_encoded)
    y_encoder = keras.Model(encoder_input, encoder_output, name='encoder')
    y_encoder.summary()

    # Establish Architecture of the Decoder
    decoder_input = keras.Input(shape=(7, 7, channel_max), name='encoded_image')
    inner_decoded = layers.Conv2DTranspose(channel_max, 3, activation='relu', padding='same')(decoder_input)
    inner_decoded = layers.UpSampling2D(2)(inner_decoded)
    inner_decoded = layers.Conv2DTranspose(int(channel_max / 2), 3, activation='relu', padding='same')(inner_decoded)
    inner_decoded = layers.UpSampling2D(2)(inner_decoded)
    decoder_output = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(inner_decoded)
    y_decoder = keras.Model(decoder_input, decoder_output, name='decoder')
    y_decoder.summary()

    # Create Full Autoencoder Model
    autoencoder_input = keras.Input(shape=(28, 28, 1), name='original_image')
    encoded_image = y_encoder(autoencoder_input)
    decoded_image = y_decoder(encoded_image)
    y_autoencoder = keras.Model(autoencoder_input, decoded_image, name='autoencoder')
    y_autoencoder.summary()

    # Compile Autoencoder
    y_autoencoder.compile(optimizer=ae_optimizer,
                            loss='mse')
    history = y_autoencoder.fit(y_train, y_train,
                    epochs=epoch_num,
                    batch_size=256,
                    validation_data=(y_val, y_val))

    # Save Model
    y_encoder.save('models/MNIST_'+ label +'_encoder.keras')
    y_decoder.save('models/MNIST_' + label +'_decoder.keras')

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/' + label +'_encoder loss')
    plt.show()
