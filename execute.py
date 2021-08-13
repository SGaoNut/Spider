import os
import sys
import time
import tensorflow as tf
import io


train_src = 'train_data/seq.data'
max_train_data_size = 50000
vocab_inp_size = 20000
enc_vocab_size = 20000
vocab_tar_size = 20000
embedding_dim = 128
units = 256
BATCH_SIZE = 128
max_length_inp, max_length_tar = 20, 20

# units 64
# ValueError: Tensor's shape (128, 768) is not compatible with supplied shape (128, 192)

def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    # print(w)
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def read_data(path, num_examples):
    input_lang, target_lang = create_dataset(path, num_examples)

    input_tensor, input_token = tokenize(input_lang)
    target_tensor, target_token = tokenize(target_lang)

    return input_tensor, input_token, target_tensor, target_token


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=enc_vocab_size, oov_token=3)
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length_inp, padding='post')

    return tensor, lang_tokenizer


input_tensor, input_token, target_tensor, target_token = read_data(train_src, max_train_data_size)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


# @tf.function
def train_step(inp, targ, targ_lang, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


checkpoint_dir = 'checkpoint'


def train(save_dir):
    checkpoint_dir = save_dir
    print("Preparing data in %s" % train_src)
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    print(steps_per_epoch)
    enc_hidden = encoder.initialize_hidden_state()
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        print("reload pretrained model")
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    checkpoint_dir = save_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()

    while True:
        start_time_epoch = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, target_token, enc_hidden)
            total_loss += batch_loss
            print(batch_loss.numpy())

        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = + steps_per_epoch
        step_time_total = (time.time() - start_time) / current_steps

        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      step_loss.numpy()))
        checkpoint.save(file_prefix=checkpoint_prefix)

        sys.stdout.flush()


def predict(sentence, model_path):
    checkpoint_dir = model_path
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    sentence = preprocess_sentence(sentence)
    inputs = [input_token.word_index.get(i, 3) for i in sentence.split(' ')]

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)

    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        if target_token.index_word[predicted_id] == 'end':
            break
        result += target_token.index_word[predicted_id] + ' '

        dec_input = tf.expand_dims([predicted_id], 0)

    return result
