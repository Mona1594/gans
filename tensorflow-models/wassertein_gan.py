import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


class config:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1
    EPOCHS = 10000
    CLIP = 0.01
    CRITIC_SIZE = 5
    BATCH_SIZE = 64
    LATENT_DIM = 100
    LEARNING_RATE = 0.00001
    LAMBDA = 10
    BETA_1 = 0
    BETA_2 = 0.9

    LOG_INTERVAL = 1000
    SAMPLE_INTERVAL = 1000


class WasserteinGAN:
    def __init__(self):
        self.image_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
        self.optimizer_ascent = tf.keras.optimizers.RMSprop(learning_rate=-config.LEARNING_RATE)
        self.optimizer_descent = tf.keras.optimizers.RMSprop(learning_rate=config.LEARNING_RATE)
        self.kernel_init = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        print("Loading Data...")
        (self.train_images, _), (_, _) = mnist.load_data()

        print("Normalizing Data...")
        self.train_images = self.train_images / 127.5 - 1.

        self.train_images = np.expand_dims(self.train_images, axis=3)
        print("Data Shape : ", self.train_images.shape)
        print()

        print("Building Generator...")
        self.generator = self.build_generator()

        print("Building Discriminator...")
        self.discriminator = self.build_discriminator()
        
        self.generator_losses = []
        self.discriminator_losses = []

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256, input_dim=config.LATENT_DIM),
            tf.keras.layers.Reshape((7, 7, 256)),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=1, padding='SAME', kernel_initializer=self.kernel_init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='SAME', kernel_initializer=self.kernel_init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='SAME', kernel_initializer=self.kernel_init),
            tf.keras.layers.Activation('tanh')
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=2, input_shape=self.image_shape, padding='SAME', kernel_initializer=self.kernel_init),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='SAME', kernel_initializer=self.kernel_init),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='SAME', kernel_initializer=self.kernel_init),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])
        return model
    
    def train_generator_step(self):
        noise = tf.random.normal((config.BATCH_SIZE, config.LATENT_DIM))
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_images, training=True)
            loss = -tf.reduce_mean(fake_pred)

        g_loss = -loss
        gradients =  tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer_descent.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return g_loss
    
    def train_discriminator_step(self):
        d_loss = []
        
        for i in range(config.CRITIC_SIZE):
            real_images = tf.convert_to_tensor(self.random_images())
            noise = tf.random.normal((config.BATCH_SIZE, config.LATENT_DIM))
            fake_images = self.generator(noise, training=False)

            with tf.GradientTape() as tape:
                real_pred = self.discriminator(real_images, training=True)
                fake_pred = self.discriminator(fake_images, training=True)
                loss = tf.reduce_mean(real_pred) - tf.reduce_mean(fake_pred)            

            d_loss.append(loss)

            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.optimizer_ascent.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        
        d_loss = tf.reduce_mean(d_loss)
        return d_loss

    @tf.function
    def train_step(self):
        # training discriminator (critic)
        d_loss = self.train_discriminator_step()

        # training generator
        g_loss = self.train_generator_step()
        
        return g_loss, d_loss

    def train(self):
        for epoch in range(config.EPOCHS):
            g_loss, d_loss = self.train_step()

            if (epoch + 1) % config.LOG_INTERVAL == 0:
                self.log_progress(epoch+1, g_loss, d_loss)

            if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
                self.sample_images(epoch+1)
            
            self.generator_losses.append(g_loss)
            self.discriminator_losses.append(d_loss)
        
        self.generate_progress_graph()

    def log_progress(self, epoch, g_loss, d_loss, accuracy=None):
        print("Epoch {}/{} :".format(epoch+1, config.EPOCHS))
        print("    [G Loss - {:.4f}]\t[D Loss - {:.4f}".format(g_loss, d_loss), end='')
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

    def random_images(self):
        indexes = np.random.randint(
            0, self.train_images.shape[0], size=config.BATCH_SIZE)
        return self.train_images[indexes]
    
    def sample_images(self, epoch):
        rows, cols = 4, 4

        noise = tf.random.normal((rows * cols, config.LATENT_DIM))

        fake_images = self.generator.predict(noise)
        fake_images = 0.5 * fake_images + 0.5

        fig, axes = plt.subplots(
            rows, cols, sharex=True, sharey=True, figsize=(10, 10))

        count = 0

        for i in range(rows):
            for j in range(cols):
                axes[i, j].imshow(fake_images[count, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
                count += 1

        plt.savefig(
            "/content/image_at_{:04d}.png".format(epoch), bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    wgan = WasserteinGAN()
    wgan.train()
