import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class config:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1
    
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 10000
    LATENT_DIM = 100
    SAMPLE_EVERY_LOOP = 1000
    LOG_EVERY_LOOP = 500
    

class Generator:
    @staticmethod
    def build():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256, input_dim=config.LATENT_DIM),
            tf.keras.layers.Reshape((7, 7, 256)),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=1, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='SAME'),
            tf.keras.layers.Activation('tanh')
        ])
        return model


class Discriminator:
    @staticmethod
    def build():
      image_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)

      model = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(32, 3, strides=2, input_shape=image_shape, padding='SAME'),
          tf.keras.layers.LeakyReLU(0.01),
          tf.keras.layers.Conv2D(64, 3, strides=2, padding='SAME'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(0.01),
          tf.keras.layers.Conv2D(128, 3, strides=2, padding='SAME'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(0.01),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Activation('sigmoid')
      ])

      return model

class GAN:
    def __init__(self):

        print("Loading Generator Model...")
        self.generator = Generator.build()

        print("Loading Discriminator Model...")
        self.discriminator = Discriminator.build()

        print("Building GAN Model...")
        self.gan = self.build()

        print("Loading Data...")
        (self.X_train, _), (_, _) = mnist.load_data()

        print("Normalizing Data...")
        self.X_train = self.X_train / 127.5 - 1.

        self.X_train = np.expand_dims(self.X_train, axis=3)
        print("Data Shape : ", self.X_train.shape)

        self.generator_losses = []
        self.discriminator_losses = []

    
    def random_images(self):
        indexes = np.random.randint(len(self.X_train), size=config.BATCH_SIZE)
        images = self.X_train[indexes]
        return images

    def sample_images(self, epoch):
        rows, cols = 4, 4
        Z = np.random.normal(0, 1, size=(rows * cols, config.LATENT_DIM))
        fake_images = self.generator.predict(Z)

        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(10, 10))

        fake_images = 0.5 * fake_images + 0.5

        count = 0

        for i in range(rows):
          for j in range(cols):
            axes[i, j].imshow(fake_images[count, :, :, 0], cmap='gray')
            axes[i, j].axis('off')
            count += 1
        
        plt.savefig("/content/image_at_{:04d}.png".format(epoch), bbox_inches='tight')
        plt.close(fig)


    def train(self):

        real_labels = np.ones((config.BATCH_SIZE, 1))
        fake_labels = np.zeros((config.BATCH_SIZE, 1))

        for epoch in range(config.EPOCHS):
          
            # getting random images from training set
            real_images = self.random_images()
            
            # generating a latent vector
            Z = np.random.normal(0, 1, size=(config.BATCH_SIZE, config.LATENT_DIM))
            
            # generating fake images
            fake_images = self.generator.predict(Z)

            # training discriminator
            d_real_loss = self.discriminator.train_on_batch(real_images, real_labels)
            d_fake_loss = self.discriminator.train_on_batch(fake_images, fake_labels)

            # getting avarage of discriminator losses
            d_loss, accuracy = 0.5 * np.add(d_fake_loss, d_real_loss)

            # training generator
            g_loss = self.gan.train_on_batch(Z, real_labels)

            self.generator_losses.append(g_loss)
            self.discriminator_losses.append(d_loss)

            if (epoch+1) % config.LOG_EVERY_LOOP == 0:
                print("Epoch {:04d} :".format(epoch+1))
                print("    Generator Loss - {:.4f}\tDiscriminator Loss - {:.4f}".format(g_loss, d_loss))
              
            if (epoch + 1) % config.SAMPLE_EVERY_LOOP == 0:
              self.sample_images(epoch+1)

    def build(self):
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False

        model = tf.keras.models.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), loss='binary_crossentropy')
        return model
    

if __name__ == "__main__":
    gan = GAN()
    gan.train()