import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class config:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1
    EPOCHS = 50000
    CLIP = 0.01
    CRITIC_SIZE = 5
    BATCH_SIZE = 64
    LATENT_DIM = 100
    LEARNING_RATE = 0.00005
    LEAKY_RELU_ALPHA = 0.01
    
    LOG_INTERVAL = 1000
    SAMPLE_INTERVAL = 1000


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)
    
    def get_config(self):
        return {'clip_value': self.clip_value}
    

class WasserteinGAN:
    def __init__(self):
        self.image_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
        self.optimizer_ascent = tf.keras.optimizers.RMSprop(learning_rate=-config.LEARNING_RATE)
        self.optimizer_descent = tf.keras.optimizers.RMSprop(learning_rate=config.LEARNING_RATE)
        self.constraint = ClipConstraint(config.CLIP)

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

        
    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256, input_dim=config.LATENT_DIM),
            tf.keras.layers.Reshape((7, 7, 256)),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=1, use_bias=False, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, use_bias=False, padding='SAME'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(1, 3, strides=2, use_bias=False,  padding='SAME'),
            tf.keras.layers.Activation('tanh')
        ])
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=2, input_shape=self.image_shape, padding='SAME', kernel_constraint=self.constraint),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='SAME', kernel_constraint=self.constraint),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='SAME', kernel_constraint=self.constraint),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, kernel_constraint=self.constraint),
            tf.keras.layers.Activation('sigmoid')
        ])
        return model
    
    @tf.function
    def train_step(self):
        d_loss = []
        g_loss = 0.

        # training discriminator (critic)
        for i in range(config.CRITIC_SIZE):
            
            real_images = tf.convert_to_tensor(self.random_images())
            Z = tf.random.normal((config.BATCH_SIZE, config.LATENT_DIM))
            
            fake_images = self.generator(Z, training=False)

            with tf.GradientTape() as tape:
                real_pred = self.discriminator(real_images, training=True)
                fake_pred = self.discriminator(fake_images, training=True)
                loss = tf.reduce_mean(real_pred) - tf.reduce_mean(fake_pred)            

            d_loss.append(loss)

            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.optimizer_ascent.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        
        d_loss = tf.reduce_mean(d_loss)

        # training generator
        Z = tf.random.normal((config.BATCH_SIZE, config.LATENT_DIM))
        with tf.GradientTape() as tape:
            fake_pred = self.discriminator(self.generator(Z, training=True), training=False)
            loss = -tf.reduce_mean(fake_pred)

        g_loss = -loss
        gradients =  tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer_descent.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss

    def train(self):
        for epoch in range(config.EPOCHS):
            d_loss, g_loss = self.train_step()
            
            if (epoch+1) % config.LOG_INTERVAL == 0:
                print("Epoch {}/{} :".format(epoch+1, config.EPOCHS))
                print("    [G Loss - {:.4f}]\t[D Loss - {:.4f}]".format(g_loss, d_loss))
              
            if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
                self.sample_images(epoch+1)
                
    def sample_images(self, epoch):
        rows, cols = 4, 4
        
        Z = tf.random.normal((rows * cols, config.LATENT_DIM))
        
        fake_images = self.generator.predict(Z)
        fake_images = 0.5 * fake_images + 0.5
        
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(10, 10))

        count = 0

        for i in range(rows):
          for j in range(cols):
            axes[i, j].imshow(fake_images[count, :, :, 0], cmap='gray')
            axes[i, j].axis('off')
            count += 1
        
        plt.savefig("/content/image_at_{:04d}.png".format(epoch), bbox_inches='tight')
        plt.close(fig)
        
    def random_images(self):
        indexes = np.random.randint(0, self.train_images.shape[0], size=config.BATCH_SIZE)
        return self.train_images[indexes]    
    def get_config(self):
        return {'clip_value': self.clip_value}
    

class WasserteinGAN:
    def __init__(self):
        self.image_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
        self.optimizer_ascent = tf.keras.optimizers.RMSprop(learning_rate=-config.LEARNING_RATE)
        self.optimizer_descent = tf.keras.optimizers.RMSprop(learning_rate=config.LEARNING_RATE)
        self.constraint = ClipConstraint(config.CLIP)

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

        
    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=config.LATENT_DIM),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(np.prod(self.image_shape)),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Reshape(self.image_shape), 
        ])
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.image_shape),
            tf.keras.layers.Dense(512, kernel_constraint=self.constraint),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(512, kernel_constraint=self.constraint),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, kernel_constraint=self.constraint),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, kernel_constraint=self.constraint),
            tf.keras.layers.Activation('sigmoid')
        ])
        return model
    
    @tf.function
    def train_step(self):
        d_loss = []
        g_loss = 0.

        # training discriminator (critic)
        for i in range(config.CRITIC_SIZE):
            
            real_images = tf.convert_to_tensor(self.random_images())
            Z = tf.random.normal((config.BATCH_SIZE, config.LATENT_DIM))
            
            fake_images = self.generator(Z, training=False)

            with tf.GradientTape() as tape:
                real_pred = self.discriminator(real_images, training=True)
                fake_pred = self.discriminator(fake_images, training=True)
                loss = tf.reduce_mean(real_pred) - tf.reduce_mean(fake_pred)            

            d_loss.append(loss)

            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.optimizer_ascent.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        
        d_loss = tf.reduce_mean(d_loss)

        # training generator
        Z = tf.random.normal((config.BATCH_SIZE, config.LATENT_DIM))
        with tf.GradientTape() as tape:
            fake_pred = self.discriminator(self.generator(Z, training=True), training=False)
            loss = -tf.reduce_mean(fake_pred)

        g_loss = -loss
        gradients =  tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer_descent.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return d_loss, g_loss

    def train(self):
        for epoch in range(config.EPOCHS):
            d_loss, g_loss = self.train_step()
            
            if (epoch+1) % config.LOG_INTERVAL == 0:
                print("Epoch {}/{} :".format(epoch+1, config.EPOCHS))
                print("    [G Loss - {:.4f}]\t[D Loss - {:.4f}]".format(g_loss, d_loss))
              
            if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
                self.sample_images(epoch+1)
                
    def sample_images(self, epoch):
        rows, cols = 4, 4
        
        Z = tf.random.normal((rows * cols, config.LATENT_DIM))
        
        fake_images = self.generator.predict(Z)
        fake_images = 0.5 * fake_images + 0.5
        
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(10, 10))

        count = 0

        for i in range(rows):
          for j in range(cols):
            axes[i, j].imshow(fake_images[count, :, :, 0], cmap='gray')
            axes[i, j].axis('off')
            count += 1
        
        plt.savefig("/content/image_at_{:04d}.png".format(epoch), bbox_inches='tight')
        plt.close(fig)
        
    def random_images(self):
        indexes = np.random.randint(0, self.train_images.shape[0], size=config.BATCH_SIZE)
        return self.train_images[indexes]
    
if __name__ == "__main__":
    wgan = WasserteinGAN()
    wgan.train()