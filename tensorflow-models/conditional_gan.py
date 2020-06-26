import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

class config:
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNELS = 1
    EPOCHS = 5000
    BATCH_SIZE = 128
    LATENT_DIM = 100
    LEARNING_RATE = 0.001
    BETA_1 = 0.9
    NUM_LABELS = 10
    
    LOG_INTERVAL = 500
    SAMPLE_INTERVAL = 1000
    
    

class ConditionalGAN:
    def __init__(self):
        
        self.image_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, config.BETA_1)
        
        print("Loading Data...")
        (self.train_images, self.train_labels), (_, _) = mnist.load_data()
        
        print("Normalizing Data...")
        self.train_images = self.train_images / 127.5 - 1.
        
        self.train_images = np.expand_dims(self.train_images, axis=3)
        print("Data Shape : ", self.train_images.shape)
        print()
        
        print("Building Generator...")
        self.generator = self.build_generator()
        
        print("Building Discriminator...")
        self.discriminator = self.build_discriminator()
        
        print("Building GAN...")
        self.gan = self.build_gan()
        
    
    def build_generator(self):
        
        noise_input = tf.keras.Input(shape=(config.LATENT_DIM,))
        label_input = tf.keras.Input(shape=(1,), dtype='int32')
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=config.LATENT_DIM),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(np.prod(self.image_shape)),
            tf.keras.layers.Activation('tanh'),
            tf.keras.layers.Reshape(self.image_shape)
        ])
        
        # creates a 100 dimensional embedding vector associated with the label
        embedding_output = tf.keras.layers.Embedding(config.NUM_LABELS, config.LATENT_DIM)(label_input)
        embedding_output = tf.keras.layers.Flatten()(embedding_output)
        print("Embedding Output Shape : ", embedding_output.output_shape)
        
        model_input = tf.keras.layers.multiply([noise_input, embedding_output])
        print("Model Input Shape : ", model_input.output_shape)
        
        fake_image = model(model_input)
        print("Fake Image Shape : ", fake_image.output_shape)
        
        return tf.keras.Model([noise_input, label_input], fake_image)

    
    def build_discriminator(self):
        
        image_input = tf.keras.Input(shape=self.image_shape)
        label_input = tf.keras.Input(shape=(1,), dtype='int32')
        
        model = tf.keras.Sequential([
            tf.keras.Dense(512, input_dim=np.prod(self.image_shape)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])
        
        embedding_output = tf.keras.layers.Embedding(config.NUM_LABELS, np.prod(self.image_shape))
        embedding_output = tf.keras.layers.Flatten()(embedding_output)
        assert embedding_output.shape == (None, 784)
        
        flat_image_input = tf.keras.layers.Flatten()(image_input)
        assert flat_image_input.shape == (None, 784)
        
        model_input = tf.keras.layers.multiply([flat_image_input, embedding_output])
        prediction = model(model_input)
        
        return tf.keras.Model([image_input, label_input], prediction)
    
    def build_gan(self):
        self.discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False
        
        noise_input = tf.keras.Input(shape=(config.LATENT_DIM,))
        label_input = tf.keras.Input(shape=(1,), dtype='int32')
        
        fake_images = self.generator([noise_input, label_input])
        
        prediction = self.discriminator([fake_images, label_input])
        
        model = tf.keras.Model([noise_input, label_input], prediction)
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy')
        
        return model
    
    def random_images_labels(self):
        indexes = np.random.randint(0, self.train_images.shape[0], size=config.BATCH_SIZE)
        images = self.train_images[indexes]
        labels = self.train_labels[indexes]
        return images, labels
    
    def sample_images(self, epoch):
        rows, cols = 2, 5
        
        Z = np.random.normal(0, 1, size=(rows * cols, config.LATENT_DIM))
        labels = np.arange(rows * cols).reshape(-1, 1)
        
        fake_images = self.generator.predict([Z, labels])
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
    
    def train(self):
        
        valid_labels = np.ones((config.BATCH_SIZE, 1))
        fake_labels = np.zeros((config.BATCH_SIZE, 1))
        
        for epoch in range(config.EPOCHS):
            real_images, real_labels = self.random_images_labels()
            
            Z = np.random.normal(0, 1, size=(config.BATCH_SIZE, config.LATENT_DIM))
            
            # generating fake images
            fake_images = self.generator.predict([Z, real_labels])
            
            # training discriminator
            d_real_loss = self.discriminator.train_on_batch([real_images, real_labels], valid_labels)
            d_fake_loss = self.discriminator.train_on_batch([fake_images, real_labels], fake_labels)
            
            d_loss, accuracy = 0.5 * np.add(d_real_loss, d_fake_loss)
            
            # training gan
            g_loss = self.gan.train_on_batch([Z, real_labels], valid_labels)
            
            if (epoch+1) % config.LOG_INTERVAL == 0:
                print("Epoch {}/{} :".format(epoch+1, config.EPOCHS))
                print("    [G Loss - {:.4f}]\t[D Loss - {:.4f} | D Acc - {:.4f}]".format(g_loss, d_loss, accuracy))
              
            if (epoch + 1) % config.SAMPLE_INTERVAL == 0:
                self.sample_images(epoch+1)
            