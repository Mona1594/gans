import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import seaborn as sns


class config:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1

    LEAKY_RELU_ALPHA = 0.3
    LEARNING_RATE = 0.001
    BETA_1 = 0.9
    BATCH_SIZE = 128
    EPOCHS = 5000
    LATENT_DIM = 100
    SAMPLE_INTERVAL = 1000
    LOG_INTERVAL = 500


class DCGAN:
    def __init__(self):

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE, beta_1=config.BETA_1)

        print("Loading Generator Model...")
        self.generator = self.build_generator()

        print("Loading Discriminator Model...")
        self.discriminator = self.build_discriminator()

        print("Building GAN Model...")
        self.gan = self.build_gan()

        print("Loading Data...")
        (self.X_train, _), (_, _) = mnist.load_data()

        print("Normalizing Data...")
        self.X_train = self.X_train / 127.5 - 1.

        self.X_train = np.expand_dims(self.X_train, axis=3)
        print("Data Shape : ", self.X_train.shape)
        print()

        self.generator_losses = []
        self.discriminator_losses = []

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256, input_dim=config.LATENT_DIM),
            tf.keras.layers.Reshape((7, 7, 256)),
            tf.keras.layers.Conv2DTranspose(
                128, 3, strides=1, use_bias=False, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(config.LEAKY_RELU_ALPHA),
            tf.keras.layers.Conv2DTranspose(
                64, 3, strides=2, use_bias=False, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(config.LEAKY_RELU_ALPHA),
            tf.keras.layers.Conv2DTranspose(
                1, 3, strides=2, use_bias=False,  padding='SAME'),
            tf.keras.layers.Activation('tanh')
        ])
        return model

    def build_discriminator(self):
        image_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, strides=2, input_shape=image_shape, padding='SAME'),
            tf.keras.layers.LeakyReLU(config.LEAKY_RELU_ALPHA),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='SAME'),
            tf.keras.layers.LeakyReLU(config.LEAKY_RELU_ALPHA),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='SAME'),
            tf.keras.layers.LeakyReLU(config.LEAKY_RELU_ALPHA),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])
        return model

    def build_gan(self):
        self.discriminator.compile(
            optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False

        model = tf.keras.models.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        return model
    
    def train_generator_step(self, Z, real_labels):
        g_loss = self.gan.train_on_batch(Z, real_labels)
        return g_loss

    def train_discriminator_step(self, Z, real_images, real_labels, fake_labels):
        # generating fake images
        fake_images = self.generator.predict(Z)

        # training discriminator
        d_real_loss = self.discriminator.train_on_batch(
            real_images, real_labels)
        d_fake_loss = self.discriminator.train_on_batch(
            fake_images, fake_labels)

        # getting avarage of discriminator losses
        d_loss, accuracy = 0.5 * np.add(d_fake_loss, d_real_loss)
        return d_loss, accuracy
    
    @tf.function
    def train_step(self):
        Z = np.random.normal(0, 1, size=(config.BATCH_SIZE, config.LATENT_DIM))
        real_images = self.random_images()
        fake_labels = np.zeros((config.BATCH_SIZE, 1))
        real_labels = np.ones((config.BATCH_SIZE, 1))
        
         # train discriminator
        d_loss, accuracy = self.train_discriminator_step(Z, real_images, real_labels, fake_labels)

        # train generator
        g_loss = self.train_generator_step(Z, real_labels)
        
        return g_loss, d_loss, accuracy

    def train(self):
        for epoch in range(config.EPOCHS):
            g_loss, d_loss, accuracy = self.train_step()

            if (epoch + 1) % config.LOG_INTERVAL == 0:
                self.log_progress(epoch+1, g_loss, d_loss, accuracy=accuracy)

            if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
                self.sample_images(epoch+1)

            self.generator_losses.append(g_loss)
            self.discriminator_losses.append(d_loss)
        
        self.generate_progress_graph()

    def random_images(self):
        indexes = np.random.randint(
            0, self.X_train.shape[0], size=config.BATCH_SIZE)
        images = self.X_train[indexes]
        return images

    def log_progress(self, epoch, g_loss, d_loss, accuracy=None):
        print("Epoch {}/{} :".format(epoch+1, config.EPOCHS))
        print(
            "    [G Loss - {:.4f}]\t[D Loss - {:.4f}".format(g_loss, d_loss), end='')
        if accuracy is not None:
            print(" | D Acc - {:.4f}]".format(accuracy))
        else:
            print("]")
            
    def generate_progress_graph(self):
        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(24, 16))
        axes[0].plot(self.generator_losses, color='purple',label='Generator Loss')
        axes[1].plot(self.discriminator_losses, color='b', label='Discriminator Loss')

        axes[0].set_title("Generator Loss")
        axes[1].set_title("Discriminator Loss")

        axes[0].set_xlabel("Epochs")
        axes[1].set_xlabel("Epochs")

        axes[0].set_ylabel("Loss")
        axes[1].set_ylabel("Loss")
        plt.savefig('/content/progress_graph.png', bbox_inches='tight')
        plt.close(fig)

    def sample_images(self, epoch):
        rows, cols = 4, 4
        Z = np.random.normal(0, 1, size=(rows * cols, config.LATENT_DIM))
        fake_images = self.generator.predict(Z)

        fig, axes = plt.subplots(
            rows, cols, sharex=True, sharey=True, figsize=(10, 10))

        fake_images = 0.5 * fake_images + 0.5

        count = 0

        for i in range(rows):
            for j in range(cols):
                axes[i, j].imshow(fake_images[count, :, :, :], cmap='gray')
                axes[i, j].axis('off')
                count += 1

        plt.savefig(
            "/content/image_at_epoch{}.png".format(epoch), bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    gan = DCGAN()
    gan.train()
