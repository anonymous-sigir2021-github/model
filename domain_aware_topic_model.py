import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from mymodel import lda_data_util as utils
from mymodel import my_data_util
import os, multiprocessing

K = tf.keras

default_gpu = 5
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % default_gpu
print('GPU:', default_gpu)

def prepare_data(repeat):

    print('repeat seed', repeat)
    np.random.seed(repeat)
    tf.random.set_seed(repeat)

    lda_dataset_path = './mymodel/data/lda_dataset.pickle'
    stopwords_path = './mymodel/data/english.txt'

    itos, stoi, lda_unlabeled_data, lda_val_unlabeled_data, lda_test_data = my_data_util.read_pickle_data(
        lda_dataset_path)
    lda_data_util = utils.lda_data(stopwords_path, itos)
    lda_unlabeled_dataset = lda_data_util.prepare_dataset(lda_unlabeled_data, 500, 20000, combine_data=True)
    lda_val_unlabeled_dataset = lda_data_util.prepare_dataset(lda_val_unlabeled_data, 500, 0, combine_data=True)
    lda_test_dataset = lda_data_util.prepare_dataset(lda_test_data, 500 * 5, shuffle=0, combine_data=False)

    return itos, stoi, lda_unlabeled_dataset, lda_val_unlabeled_dataset, lda_test_dataset


class my_multi_domain_encoder_model(K.Model):

    def __init__(self, ndomain, n_input, n_spec_topic, n_share_topic, activation, use_bias=True, *args, **kwargs):
        super().__init__(name='myencoder', *args, **kwargs)

        self.ndomain = ndomain
        self.n_input = n_input
        self.n_spec_topic = n_spec_topic
        self.n_share_topic = n_share_topic
        if activation is None:
            self.activation = None
        elif activation == 'relu':
            self.activation = K.activations.relu
        elif activation == 'tanh':
            self.activation = K.activations.tanh
        else:
            print("Error activation")
        self.use_bias = use_bias
        self.domain_specific_kernel = self.add_weight(
            'myencoder_spec_k',
            [ndomain, n_input, n_spec_topic],
            dtype='float32',
            initializer=K.initializers.GlorotUniform(),
        )
        if self.use_bias:
            self.domain_specific_bias = self.add_weight(
                'myencoder_spec_b',
                [ndomain, 1, n_spec_topic],
                dtype='float32',
                initializer=K.initializers.Zeros(),
            )
        self.domain_share_layer = K.layers.Dense(
            n_share_topic,
            activation,
            use_bias,
            name='myencoder_sublayer'
        )

    def call(self, inputs, training=None, mask=None):
        re_inputs = tf.reshape(inputs, [self.ndomain, -1, self.n_input])
        spec_output = re_inputs @ self.domain_specific_kernel
        if self.use_bias:
            spec_output += self.domain_specific_bias
        if self.activation is not None:
            spec_output = self.activation(spec_output)

        re_spec_output = tf.reshape(spec_output, [-1, self.n_spec_topic])
        share_output = self.domain_share_layer(inputs)

        output = tf.concat([re_spec_output, share_output], axis=1)
        return output


class my_multi_domain_decoder_model(K.Model):

    def __init__(self, ndomain, n_spec_topic, n_share_topic, n_vocab, *args, **kwargs):
        super().__init__(name='mydecoder', *args, **kwargs)

        self.ndomain = ndomain
        self.n_spec_topic = n_spec_topic
        self.n_share_topic = n_share_topic
        self.n_vocab = n_vocab
        self.domain_specific_kernel = self.add_weight(
            'mydecoder_spec_k',
            [ndomain, n_spec_topic, n_vocab],
            dtype='float32',
            initializer=K.initializers.GlorotUniform(),
        )
        self.domain_share_layer = K.layers.Dense(
            n_vocab,
            activation=None,
            use_bias=False,
            name='mydecoder_sublayer'
        )

    def call(self, inputs, training=None, mask=None):
        spec, share = tf.split(inputs, [self.n_spec_topic, self.n_share_topic], axis=-1)
        re_spec = tf.reshape(spec, [self.ndomain, -1, self.n_spec_topic])
        spec_output = re_spec @ self.domain_specific_kernel

        re_spec_output = tf.reshape(spec_output, [-1, self.n_vocab])
        share_output = self.domain_share_layer(share)

        output = re_spec_output + share_output
        return output

class RSVI(K.Model):

    def __init__(self,
                 vocab_size,
                 n_topic,
                 non_linearity,
                 dir_prior,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.n_topic = n_topic
        self.non_linearity = non_linearity

        self.dir_prior = dir_prior

        self.x = K.layers.InputLayer(
            [vocab_size],
            None,
            name='input',
            dtype='float32',
        )
        self.n_hidden = 100
        self.keep_prob = .75
        self.min_alpha = .00001
        self.B = 10
        # encoder

        self.encoder = K.layers.Dense(
            self.n_hidden,
            self.non_linearity,
            name='encoder'
        )
        self.encoder_mean = my_multi_domain_encoder_model(5, self.n_hidden, self.n_topic[0], self.n_topic[1], None,
                                                          True)
        self.encoder_dropout = K.layers.Dropout(1 - self.keep_prob, name='encoder_dropout')
        self.encoder_batchnorm = K.layers.BatchNormalization(name='encoder_batnorm')

        self.decoder = my_multi_domain_decoder_model(5, self.n_topic[0], self.n_topic[1], self.vocab_size)
        self.decoder_batchnorm = K.layers.BatchNormalization(name='decoder_batnorm')

        self.encoder_vars = None
        self.decoder_vars = None

        self.enc_kl_m = K.metrics.Mean('enc_kl')
        self.enc_reconst_m = K.metrics.Mean('enc_reconst')
        self.dec_kl_m = K.metrics.Mean('dec_kl')
        self.dec_reconst_m = K.metrics.Mean('dec_reconst')

    @property
    def metrics(self):
        return [self.enc_kl_m, self.enc_reconst_m, self.dec_kl_m, self.dec_reconst_m]

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_states()

    def get_metrics_result(self):
        return {m.name: m.result().numpy() for m in self.metrics}

    def get_vars(self):
        if self.encoder_vars is None:
            encoder_vars = []
            all_vars = self.trainable_variables
            for var in all_vars:
                if 'encoder' in var.name:
                    encoder_vars.append(var)
            self.encoder_vars = encoder_vars
        if self.decoder_vars is None:
            decoder_vars = []
            all_vars = self.trainable_variables
            for var in all_vars:
                if 'decoder' in var.name:
                    decoder_vars.append(var)
            self.decoder_vars = decoder_vars
        return self.encoder_vars, self.decoder_vars

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = tf.sparse.to_dense(x)
        batch_size, n_vocab = tf.shape(x)
        self.raw_input = x

        x = self.x(x)

        #######################
        # Encoder
        ######################
        self.enc_vec = self.encoder(x)
        self.enc_vec = self.encoder_dropout(self.enc_vec, training=training)
        self.enc_vec_mean = self.encoder_mean(self.enc_vec)
        self.mean = self.encoder_batchnorm(self.enc_vec_mean, training=training)
        self.alpha = tf.maximum(self.min_alpha, tf.math.log(1. + tf.exp(self.mean)))

        self.prior = tf.ones((batch_size, self.n_topic[0] + self.n_topic[1]), dtype=tf.float32,
                             name='prior') * self.dir_prior

        ######################
        ###decoder
        ######################
        gam = tf.squeeze(tf.random.gamma(shape=(1,), alpha=self.alpha + self.B))
        _tmp_alpha = self.alpha + self.B
        eps = tf.sqrt(9. * _tmp_alpha - 3.) * (tf.pow(gam / (_tmp_alpha - 1. / 3.), 1. / 3.) - 1.)
        eps = tf.stop_gradient(eps)
        u = tf.random.uniform((self.B, batch_size, self.n_topic[0] + self.n_topic[1]))

        self.doc_vec = self.gamma_h_boosted(eps, u, self.alpha, self.B)
        self.doc_vec = gam / tf.reduce_sum(gam, 1, keepdims=True)

        _decode_norm = self.decoder_batchnorm(self.decoder(self.doc_vec), training=training)
        self.logits = tf.nn.log_softmax(_decode_norm)

        gammas = self.gamma_h_boosted(eps, u, self.alpha, self.B)
        self.doc_vec2 = gammas / tf.reduce_sum(gammas, 1, keepdims=True)
        _decode_norm2 = self.decoder_batchnorm(self.decoder(self.doc_vec2), training=training)
        self.logits2 = tf.nn.log_softmax(_decode_norm2)

        return self.alpha, self.prior, self.doc_vec, self.logits, self.doc_vec2, self.logits2

    def gamma_h_boosted(self, epsilon, u, alpha, model_B):
        """
        Reparameterization for gamma rejection sampler with shape augmentation.
        """
        # B = u.shape.dims[0] #u has shape of alpha plus one dimension for B
        B = self.B
        K = alpha.shape[1]  # (batch_size,K)
        r = tf.range(B, dtype='float32')
        rm = tf.reshape(r, [-1, 1, 1])  # dim Bx1x1
        alpha_vec = tf.reshape(tf.tile(alpha, (B, 1)), (model_B, -1, K)) + rm  # dim BxBSxK + dim Bx1
        u_pow = tf.pow(u, 1. / alpha_vec) + 1e-10

        b = alpha + B - 1. / 3.
        c = 1. / tf.sqrt(9. * b)
        v = 1. + epsilon * c
        gammah = b * (v ** 3)
        return tf.reduce_prod(u_pow, axis=0) * gammah

    def calculate_loss(self):
        self.recons_loss = -tf.reduce_sum(tf.multiply(self.logits, self.raw_input), 1)
        dir1 = tfp.distributions.Dirichlet(self.prior)
        dir2 = tfp.distributions.Dirichlet(self.alpha)
        self.kld = dir2.log_prob(self.doc_vec) - dir1.log_prob(self.doc_vec)

        self.recons_loss2 = -tf.reduce_sum(tf.multiply(self.logits2, self.raw_input), 1)
        self.kld2 = tfp.distributions.Dirichlet(self.alpha).log_prob(self.doc_vec2) - tfp.distributions.Dirichlet(
            self.prior).log_prob(self.doc_vec2)

        return self.kld, self.recons_loss, self.kld2, self.recons_loss2


    def evaluate(self, data):
        self.call(data, training=False)  # Forward pass
        self.calculate_loss()
        objective = self.recons_loss + self.kld
        return objective.numpy()

    def train_step(self, data, warm_up):
        x = data
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.call(x, training=True)  # Forward pass
            self.calculate_loss()
            self.objective = self.recons_loss + warm_up * self.kld

        enc_vars, dec_vars = self.get_vars()

        dec_grads = tape.gradient(self.objective, dec_vars)

        kl_grad = tape.gradient(self.kld2, enc_vars)
        g_rep = tape.gradient(self.recons_loss2, enc_vars)

        enc_grads = [g_r + warm_up * g_e for g_r, g_e in zip(g_rep, kl_grad)][:4]

        self.optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optimizer.apply_gradients(zip(dec_grads, dec_vars))
        self.enc_kl_m.update_state(self.kld2)
        self.enc_reconst_m.update_state(self.recons_loss2)
        self.dec_kl_m.update_state(self.kld)
        self.dec_reconst_m.update_state(self.recons_loss)
        del tape
        return {m.name: m.result().numpy() for m in self.metrics}

    def get_topic_proportion(self, sparse_tensor):
        alpha, prior, doc_vec, logits, doc_vec2, logits2 = self.call(sparse_tensor, training=False)
        return alpha.numpy(), doc_vec.numpy(), doc_vec2.numpy()


def train(model: RSVI,
          train_dataset,
          dev_dataset,
          alternate_epochs=1,  # 10
          warm_up_period=100,
          repeat=0):
    warm_up = 0

    early_stopping_iters = 10
    no_improvement_iters = 0
    stopped = False
    epoch = -1

    train_dataset = train_dataset.prefetch(20)
    dev_dataset = dev_dataset.prefetch(20)
    for ___ in range(warm_up_period):
        epoch += 1
        print(epoch)
        if warm_up < 1.:
            warm_up += 1. / warm_up_period
        else:
            warm_up = 1.
        for i in range(alternate_epochs):
            model.reset_metrics()
            for data in train_dataset:
                rs = model.train_step(data, warm_up)
            print(model.get_metrics_result())

    best_losses = 1e20
    while not stopped:
        epoch += 1
        warm_up = 1.
        print(epoch)

        for i in range(alternate_epochs):

            model.reset_metrics()
            for data in train_dataset:
                rs = model.train_step(data, warm_up)
            print(model.get_metrics_result())
            dev_loss = []
            for data in dev_dataset:
                rs = model.evaluate(data)
                dev_loss.extend(rs)
            dev_loss = np.mean(dev_loss)
            print(dev_loss)
            if dev_loss < best_losses:
                print('improve', dev_loss)
                no_improvement_iters = 0
                best_losses = dev_loss
                model.save_weights('./mymodel/lda_model/lda_domain_alldata_%d_%d_%d' % (
                model.n_topic[0], model.n_topic[1], repeat))
            else:
                no_improvement_iters += 1
                print('no improvement', no_improvement_iters)
                if no_improvement_iters >= early_stopping_iters:
                    stopped = True


def main(n_topic1, n_topic2, repeat=0):
    n_topic = [n_topic1, n_topic2]

    itos, stoi, lda_unlabeled_dataset, lda_val_unlabeled_dataset, lda_test_dataset = prepare_data(repeat)

    non_linearity = 'relu'
    dir_prior = 0.01
    lexicon = itos
    vocab_size = len(lexicon)

    rsvi = RSVI(
        vocab_size=vocab_size,
        n_topic=n_topic,
        non_linearity=non_linearity,
        dir_prior=dir_prior,
    )
    rsvi.compile('adam')

    train(rsvi, lda_unlabeled_dataset, lda_val_unlabeled_dataset,
          warm_up_period=100, repeat=repeat)


def create_lda_data_score(n_topic, repeat=0):

    lda_dataset_path = './mymodel/data/lda_dataset.pickle'
    stopwords_path = './mymodel/data/english.txt'

    itos, stoi, lda_unlabeled_data, lda_val_unlabeled_data, lda_test_data = my_data_util.read_pickle_data(
        lda_dataset_path)
    lda_data_util = utils.lda_data(stopwords_path, itos)
    n_domains = len(lda_test_data)
    lda_unlabeled_dataset = lda_data_util.prepare_eval_dataset(lda_unlabeled_data, 500)
    lda_test_dataset = lda_data_util.prepare_eval_dataset(lda_test_data, 500)

    lda_unlabeled_dataset = lda_unlabeled_dataset.prefetch(20)
    lda_test_dataset = lda_test_dataset.prefetch(20)

    lexicon = itos
    vocab_size = len(lexicon)
    non_linearity = 'relu'
    dir_prior = 0.01
    model = RSVI(
        vocab_size=vocab_size,
        n_topic=n_topic,
        non_linearity=non_linearity,
        dir_prior=dir_prior,
    )
    model.load_weights('./mymodel/lda_model/lda_domain_alldata_%d_%d_%d' % (
    model.n_topic[0], model.n_topic[1], repeat))

    domain_topic_proportions = [model.get_topic_proportion(_data) for _data in lda_unlabeled_dataset]
    domain_topic_proportions = [
        np.concatenate([np.reshape(data[_idx], [n_domains, -1, np.sum(n_topic)]) for data in domain_topic_proportions],
                       axis=1) for _idx in range(3)]
    domain_topic_proportions = [[domain_topic_proportions[n_output][domain] for n_output in range(3)] for domain in
                                range(n_domains)]
    domain_topic_proportions = [(alpha[:len(_l)], vec1[:len(_l)], vec2[:len(_l)]) for (alpha, vec1, vec2), _l in
                                zip(domain_topic_proportions, lda_unlabeled_data)]
    domain_topic_proportions = [(alpha / np.sum(alpha, axis=-1, keepdims=True), vec1, vec2) for alpha, vec1, vec2 in
                                domain_topic_proportions]

    target_topic_proportions = [model.get_topic_proportion(_data) for _data in lda_test_dataset]
    target_topic_proportions = [
        np.concatenate([np.reshape(data[_idx], [n_domains, -1, np.sum(n_topic)]) for data in target_topic_proportions],
                       axis=1) for _idx in range(3)]
    target_topic_proportions = [[target_topic_proportions[n_output][domain] for n_output in range(3)] for domain in
                                range(n_domains)]
    target_topic_proportions = [(alpha[:len(_l)], vec1[:len(_l)], vec2[:len(_l)]) for (alpha, vec1, vec2), _l in
                                zip(target_topic_proportions, lda_test_data)]
    target_topic_proportions = [(alpha / np.sum(alpha, axis=-1, keepdims=True), vec1, vec2) for alpha, vec1, vec2 in
                                target_topic_proportions]

    lda_score_path = './mymodel/lda_model/lda_data_score_%d_%d_%d.pickle' % (
    n_topic[0], n_topic[1], repeat)
    my_data_util.save_pickle_data((domain_topic_proportions, target_topic_proportions), lda_score_path)


def batch_run():
    n_repeat = 10
    for repeat in range(n_repeat):
        n_topic = [20, 40]
        print(n_topic)
        p = multiprocessing.Process(target=main, args=(n_topic[0], n_topic[1], repeat))
        p.start()
        p.join()


if __name__ == '__main__':
    batch_run()
    for repeat in range(10):
        create_lda_data_score([20, 40], repeat)
