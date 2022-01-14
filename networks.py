from layers import *
##################################################################################
# Generator Networks
##################################################################################
class Generator(tf.keras.Model):
    def __init__(self, g_params, **kwargs):
        super(Generator, self).__init__(**kwargs)

        self.z_dim = g_params['z_dim']
        self.w_dim = g_params['w_dim']
        self.labels_dim = g_params['labels_dim']
        self.n_mapping = g_params['n_mapping']
        self.resolutions = g_params['resolutions']
        self.featuremaps = g_params['featuremaps']
        self.w_ema_decay = g_params['w_ema_decay']
        self.style_mixing_prob = g_params['style_mixing_prob']

        self.n_broadcast = len(self.resolutions) * 2
        self.mixing_layer_indices = np.arange(self.n_broadcast)[np.newaxis, :, np.newaxis]

        self.g_mapping = Mapping(self.w_dim, self.labels_dim, self.n_mapping, name='g_mapping')
        self.broadcast = tf.keras.layers.Lambda(lambda x: tf.tile(x[:, np.newaxis], [1, self.n_broadcast, 1]))
        self.synthesis = Synthesis(self.resolutions, self.featuremaps, name='g_synthesis')



    def build(self, input_shape):
        # w_avg
        self.w_avg = tf.Variable(tf.zeros(shape=[self.w_dim], dtype=tf.float32), name='w_avg', trainable=False,
                                 synchronization=tf.VariableSynchronization.ON_READ,
                                 aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    @tf.function
    def set_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        for cw, sw in zip(self.weights, src_net.weights):
            assert sw.shape == cw.shape

            if 'w_avg' in cw.name:
                cw.assign(lerp(sw, cw, beta_nontrainable))
            else:
                cw.assign(lerp(sw, cw, beta))
        return

    def update_moving_average_of_w(self, w_broadcasted):
        # compute average of current w
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)

        # compute moving average of w and update(assign) w_avg
        update_w_avg = lerp(batch_avg, self.w_avg, self.w_ema_decay)

        return update_w_avg

    def style_mixing_regularization(self, latents1, labels, w_broadcasted1):
        # get another w and broadcast it
        latents2 = tf.random.normal(shape=tf.shape(latents1), dtype=tf.float32)
        dlatents2 = self.g_mapping([latents2, labels])
        w_broadcasted2 = self.broadcast(dlatents2)


        # find mixing limit index
        if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
            mixing_cutoff_index = tf.random.uniform([], 1, self.n_broadcast, dtype=tf.int32)
        else:
            mixing_cutoff_index = tf.constant(self.n_broadcast, dtype=tf.int32)

        # mix it
        mixed_w_broadcasted = tf.where(
            condition=tf.broadcast_to(self.mixing_layer_indices < mixing_cutoff_index, tf.shape(w_broadcasted1)),
            x=w_broadcasted1,
            y=w_broadcasted2)

        return mixed_w_broadcasted

    def truncation_trick(self, w_broadcasted, truncation_cutoff, truncation_psi):
        ones = tf.ones_like(self.mixing_layer_indices, dtype=tf.float32)
        tpsi = ones * truncation_psi

        if truncation_cutoff is None:
            truncation_coefs = tpsi
        else:
            indices = tf.range(self.n_broadcast)
            truncation_coefs = tf.where(condition=tf.less(indices, truncation_cutoff), x=tpsi, y=ones)

        truncated_w_broadcasted = lerp(self.w_avg, w_broadcasted, truncation_coefs)

        return truncated_w_broadcasted

    def call(self, inputs, truncation_cutoff=None, truncation_psi=1.0, shift_h=0, shift_w=0, training=None, mapping=True, mask=None):
        latents, labels = inputs

        if mapping:
            dlatents = self.g_mapping([latents, labels])
            w_broadcasted = self.broadcast(dlatents)

            if training:
                self.w_avg.assign(self.update_moving_average_of_w(w_broadcasted))
                w_broadcasted = self.style_mixing_regularization(latents, labels, w_broadcasted)

            if not training:
                w_broadcasted = self.truncation_trick(w_broadcasted, truncation_cutoff, truncation_psi)

        else:
            w_broadcasted = latents

        image_out = self.synthesis(w_broadcasted, shift_h=shift_h, shift_w=shift_w)

        return image_out, w_broadcasted

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)

        # shape_latents, shape_labels = input_shape
        return input_shape[0][0], 3, self.resolutions[-1], self.resolutions[-1]


##################################################################################
# Discriminator Networks
##################################################################################
class Discriminator(tf.keras.Model):
    def __init__(self, d_params, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        # discriminator's (resolutions and featuremaps) are reversed against generator's
        self.labels_dim = d_params['labels_dim']
        self.r_resolutions = d_params['resolutions'][::-1]
        self.r_featuremaps = d_params['featuremaps'][::-1]

        # stack discriminator blocks
        res0, n_f0 = self.r_resolutions[0], self.r_featuremaps[0]
        self.initial_fromrgb = FromRGB(fmaps=n_f0, name='{:d}x{:d}/FromRGB'.format(res0, res0))
        self.blocks = []

        for index, (res0, n_f0) in enumerate(zip(self.r_resolutions[:-1], self.r_featuremaps[:-1])):
            n_f1 = self.r_featuremaps[index + 1]
            self.blocks.append(DiscriminatorBlock(n_f0=n_f0, n_f1=n_f1, name='{:d}x{:d}'.format(res0, res0)))

        # set last discriminator block
        res = self.r_resolutions[-1]
        n_f0, n_f1 = self.r_featuremaps[-2], self.r_featuremaps[-1]
        self.last_block = DiscriminatorLastBlock(n_f0, n_f1, name='{:d}x{:d}'.format(res, res))

        # set last dense layer
        self.last_dense = Dense(max(self.labels_dim, 1), gain=1.0, lrmul=1.0, name='last_dense')
        self.last_bias = BiasAct(lrmul=1.0, act='linear', name='last_bias')



    # @ tf.function
    def call(self, inputs, training=None, mask=None):
        images, labels = inputs

        x = self.initial_fromrgb(images)
        for block in self.blocks:
            x = block(x)

        x = self.last_block(x)

        logit = self.last_dense(x)
        logit = self.last_bias(logit)

        if self.labels_dim > 0:
            logit = tf.reduce_sum(logit * labels, axis=1, keepdims=True)

        scores_out = logit

        return scores_out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], max(self.labels_dim, 1)

##################################################################################
# Mapping Networks
##################################################################################
class Mapping(tf.keras.layers.Layer):
    def __init__(self, w_dim, labels_dim, n_mapping, **kwargs):
        super(Mapping, self).__init__(**kwargs)
        self.w_dim = w_dim
        self.labels_dim = labels_dim
        self.n_mapping = n_mapping
        self.gain = 1.0
        self.lrmul = 0.01

        if self.labels_dim > 0:
            self.labels_embedding = LabelEmbedding(embed_dim=self.w_dim, name='labels_embedding')

        self.normalize = tf.keras.layers.Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8))

        self.dense_layers = []
        self.bias_act_layers = []

        for ii in range(self.n_mapping):
            self.dense_layers.append(Dense(w_dim, gain=self.gain, lrmul=self.lrmul, name='dense_{:d}'.format(ii)))
            self.bias_act_layers.append(BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_{:d}'.format(ii)))

    def call(self, inputs, training=None, mask=None):
        latents, labels = inputs
        x = latents

        # embed label if any
        if self.labels_dim > 0:
            y = self.labels_embedding(labels)
            x = tf.concat([x, y], axis=1)

        # normalize inputs
        x = self.normalize(x)

        # apply mapping blocks
        for dense, apply_bias_act in zip(self.dense_layers, self.bias_act_layers):
            x = dense(x)
            x = apply_bias_act(x)

        return x

    # def get_config(self):
    #     config = super(Mapping, self).get_config()
    #     config.update({
    #         'w_dim': self.w_dim,
    #         'labels_dim': self.labels_dim,
    #         'n_mapping': self.n_mapping,
    #         'gain': self.gain,
    #         'lrmul': self.lrmul,
    #     })
    #     return config
