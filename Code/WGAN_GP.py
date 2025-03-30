import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Parameters
SEQ_LENGTH = 
AMINO_ACID_SIZE = 20
NOISE_DIM = 300
BATCH_SIZE = 32
EPOCHS = 2000
AMINO_ACID_MAPPING = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}

# 1. Prepare Data
def prepare_data_from_csv(file_path, seq_length, amino_acid_mapping):
    df = pd.read_csv(file_path)
    sequences = df['Sequence'].tolist()
    data = []
    for seq in sequences:
        one_hot = np.zeros((seq_length, AMINO_ACID_SIZE))
        for i, aa in enumerate(seq):
            if i < seq_length and aa in amino_acid_mapping:
                one_hot[i, amino_acid_mapping[aa]] = 1
        data.append(one_hot.flatten())
    return np.array(data)

# 2. Build Generator
def build_generator():
    input_noise = Input(shape=(NOISE_DIM,))
    x = Dense(512, activation='leaky_relu')(input_noise)
    x = Dense(1024, activation='leaky_relu')(x)
    x = Dense(SEQ_LENGTH * AMINO_ACID_SIZE, activation='tanh')(x)
    output = Reshape((SEQ_LENGTH * AMINO_ACID_SIZE,))(x)
    generator = Model(input_noise, output, name="Generator")
    return generator

# 3. Build Critic
def build_critic():
    input_data = Input(shape=(SEQ_LENGTH * AMINO_ACID_SIZE,))
    x = Dense(512, activation='relu')(input_data)
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='linear')(x)  # no sigmoid
    critic = Model(input_data, output, name="Critic")
    return critic

# 4. Gradient Penalty
def gradient_penalty(critic, real_data, fake_data):
    batch_size = tf.shape(real_data)[0]
    alpha = tf.random.uniform((batch_size, 1), 0.0, 1.0)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        validity_interpolated = critic(interpolated)
    gradients = tape.gradient(validity_interpolated, [interpolated])[0]
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
    return penalty

# 5. WGAN-GP Training Class
class WGAN_GP:
    def __init__(self, generator, critic, noise_dim, gp_weight=10):
        self.generator = generator
        self.critic = critic
        self.noise_dim = noise_dim
        self.gp_weight = gp_weight

        self.critic_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    def train_critic(self, real_data, batch_size):
        noise = tf.random.normal((batch_size, self.noise_dim))
        fake_data = self.generator(noise, training=True)
        
        with tf.GradientTape() as tape:
            real_validity = self.critic(real_data, training=True)
            fake_validity = self.critic(fake_data, training=True)
            gp = gradient_penalty(self.critic, real_data, fake_data)
            critic_loss = tf.reduce_mean(fake_validity) - tf.reduce_mean(real_validity) + self.gp_weight * gp
        
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        return critic_loss

    def train_generator(self, batch_size):
        noise = tf.random.normal((batch_size, self.noise_dim))
        
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_validity = self.critic(fake_data, training=True)
            generator_loss = -tf.reduce_mean(fake_validity)
        
        generator_gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        return generator_loss

    def train(self, real_data, epochs, batch_size, n_critic=5):
        for epoch in range(epochs):
            for _ in range(n_critic):
                idx = np.random.randint(0, real_data.shape[0], batch_size)
                real_batch = real_data[idx]
                critic_loss = self.train_critic(real_batch, batch_size)
            
            generator_loss = self.train_generator(batch_size)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Critic Loss: {critic_loss:.4f}, Generator Loss: {generator_loss:.4f}")

# 6. Load Data
input_file = 'Original.csv'
real_data = prepare_data_from_csv(input_file, SEQ_LENGTH, AMINO_ACID_MAPPING)

# 7. Build Models
generator = build_generator()
critic = build_critic()
wgan_gp = WGAN_GP(generator, critic, NOISE_DIM)

# 8. Train WGAN-GP
wgan_gp.train(real_data, EPOCHS, BATCH_SIZE, n_critic=5)

# 9. Generate New Sequences
def decode_sequence(one_hot_encoded, seq_length, amino_acid_mapping):
    reverse_mapping = {v: k for k, v in amino_acid_mapping.items()}
    decoded_sequence = ""
    for i in range(seq_length):
        amino_acid_index = np.argmax(one_hot_encoded[i * AMINO_ACID_SIZE:(i + 1) * AMINO_ACID_SIZE])
        decoded_sequence += reverse_mapping.get(amino_acid_index, '-')
    return decoded_sequence

num_sequences = 
noise = tf.random.normal((num_sequences, NOISE_DIM))
generated_data = generator.predict(noise)
decoded_sequences = [decode_sequence(seq, SEQ_LENGTH, AMINO_ACID_MAPPING) for seq in generated_data]

# Save to CSV
output_file = 'generated.csv'
df = pd.DataFrame(decoded_sequences, columns=['Sequence'])
df.to_csv(output_file, index=False)
print(f"Generated sequences saved to {output_file}")
