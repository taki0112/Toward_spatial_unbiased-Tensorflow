from ops import *

##################################################################################
# Synthesis Layers
##################################################################################
class Synthesis(tf.keras.layers.Layer):
    def __init__(self, resolutions, featuremaps, name, **kwargs):
        super(Synthesis, self).__init__(name=name, **kwargs)
        self.resolutions = resolutions
        self.featuremaps = featuremaps

        self.k, self.pad0, self.pad1 = compute_paddings([1, 3, 3, 1], None, up=True, down=False, is_conv=False)

        # initial layer
        res, n_f = resolutions[0], featuremaps[0]
        self.img_size = resolutions[-1]
        self.log_size = int(np.log2(self.img_size))

        self.shift_h_dict = {4: 0}
        self.shift_w_dict = {4: 0}
        for i in range(3, self.log_size + 1):
            self.shift_h_dict[2 ** i] = 0
            self.shift_w_dict[2 ** i] = 0

        self.initial_block = SynthesisConstBlock(fmaps=n_f, name='{:d}x{:d}/const'.format(res, res))
        self.initial_torgb = ToRGB(in_ch=n_f, name='{:d}x{:d}/ToRGB'.format(res, res))

        # stack generator block with lerp block
        prev_n_f = n_f
        self.blocks = []
        self.torgbs = []

        for res, n_f in zip(self.resolutions[1:], self.featuremaps[1:]):
            self.blocks.append(SynthesisBlock(in_ch=prev_n_f, fmaps=n_f, res=res,
                                              name='{:d}x{:d}/block'.format(res, res)))
            self.torgbs.append(ToRGB(in_ch=n_f,  name='{:d}x{:d}/ToRGB'.format(res, res)))
            prev_n_f = n_f

    def call(self, inputs, shift_h=0, shift_w=0, training=None, mask=None):
        ##### positional encoding #####
        # continuous roll
        if shift_h:
            for i in range(2, self.log_size + 1):
                self.shift_h_dict[2 ** i] = shift_h / (self.img_size // (2 ** i))
        if shift_w:
            for i in range(2, self.log_size + 1):
                self.shift_w_dict[2 ** i] = shift_w / (self.img_size // (2 ** i))

        w_broadcasted = inputs

        # initial layer
        w0, w1 = w_broadcasted[:, 0], w_broadcasted[:, 1]

        x = self.initial_block([w_broadcasted, w0], shift_h_dict=self.shift_h_dict, shift_w_dict=self.shift_w_dict)
        y = self.initial_torgb([x, w1])

        layer_index = 1
        for block, torgb in zip(self.blocks, self.torgbs):
            w0 = w_broadcasted[:, layer_index]
            w1 = w_broadcasted[:, layer_index + 1]
            w2 = w_broadcasted[:, layer_index + 2]

            x = block([x, w0, w1],  shift_h_dict=self.shift_h_dict, shift_w_dict=self.shift_w_dict)
            y = upsample_2d(y, self.pad0, self.pad1, self.k)
            y = y + torgb([x, w2])

            layer_index += 2

        images_out = y

        return images_out

    # def get_config(self):
    #     config = super(Synthesis, self).get_config()
    #     config.update({
    #         'resolutions': self.resolutions,
    #         'featuremaps': self.featuremaps,
    #         'k': self.k,
    #         'pad0': self.pad0,
    #         'pad1': self.pad1,
    #     })
    #     return config


class SynthesisConstBlock(tf.keras.layers.Layer):
    def __init__(self, fmaps, **kwargs):
        super(SynthesisConstBlock, self).__init__(**kwargs)
        self.res = 4
        self.fmaps = fmaps
        self.gain = 1.0
        self.lrmul = 1.0

        # conv block
        self.conv = ModulatedConv2D(fmaps=self.fmaps, style_fmaps=self.fmaps, kernel=3, up=False, down=False,
                                    demodulate=True, resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul,
                                    fused_modconv=True, name='conv')
        self.apply_noise = Noise(name='noise')
        self.apply_bias_act = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias')

        self.pes_start = PE2dStart(512, 4, 4, scale=1.0)

    # def build(self, input_shape):
    #     # starting const variable
    #     # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
    #     const_init = tf.random.normal(shape=(1, self.fmaps, self.res, self.res), mean=0.0, stddev=1.0)
    #     self.const = tf.Variable(const_init, name='const', trainable=True)

    def call(self, inputs, shift_h_dict=None, shift_w_dict=None, training=None, mask=None):
        w_broadcasted, w0 = inputs
        batch_size = tf.shape(w0)[0]

        # const block
        # x = tf.tile(self.const, [batch_size, 1, 1, 1])
        x = self.pes_start(w_broadcasted, shift_h_dict[4], shift_w_dict[4])

        # conv block
        x = self.conv([x, w0])
        x = self.apply_noise(x)
        x = self.apply_bias_act(x)
        return x


class SynthesisBlock(tf.keras.layers.Layer):
    def __init__(self, in_ch, fmaps, res, **kwargs):
        super(SynthesisBlock, self).__init__(**kwargs)
        self.in_ch = in_ch
        self.fmaps = fmaps
        self.gain = 1.0
        self.lrmul = 1.0
        self.res = res

        # conv0 up
        self.conv_0 = ModulatedConv2D(fmaps=self.fmaps, style_fmaps=self.in_ch, kernel=3, up=True, down=False,
                                      demodulate=True, resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul,
                                      fused_modconv=True, name='conv_0')
        self.apply_noise_0 = Noise(name='noise_0')
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_0')

        self.pes = PE2d(channel=fmaps, height=res, width=res, scale=1.0)

        # conv block
        self.conv_1 = ModulatedConv2D(fmaps=self.fmaps, style_fmaps=self.fmaps, kernel=3, up=False, down=False,
                                      demodulate=True, resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul,
                                      fused_modconv=True, name='conv_1')
        self.apply_noise_1 = Noise(name='noise_1')
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_1')

    def call(self, inputs, shift_h_dict=None, shift_w_dict=None, training=None, mask=None):
        x, w0, w1 = inputs

        # conv0 up
        x = self.conv_0([x, w0])
        x = self.apply_noise_0(x)
        x = self.apply_bias_act_0(x)

        # pse
        x = self.pes(x, shift_h=shift_h_dict[self.res], shift_w=shift_w_dict[self.res])

        # conv block
        x = self.conv_1([x, w1])
        x = self.apply_noise_1(x)
        x = self.apply_bias_act_1(x)

        return x

    # def get_config(self):
    #     config = super(SynthesisBlock, self).get_config()
    #     config.update({
    #         'in_ch': self.in_ch,
    #         'res': self.res,
    #         'fmaps': self.fmaps,
    #         'gain': self.gain,
    #         'lrmul': self.lrmul,
    #     })
    #     return config

##################################################################################
# Discriminator Layers
##################################################################################
class DiscriminatorBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, **kwargs):
        super(DiscriminatorBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1
        self.resnet_scale = 1. / tf.sqrt(2.)

        # conv_0
        self.conv_0 = Conv2D(fmaps=self.n_f0, kernel=3, up=False, down=False,
                             resample_kernel=None, gain=self.gain, lrmul=self.lrmul, name='conv_0')
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_0')

        # conv_1 down
        self.conv_1 = Conv2D(fmaps=self.n_f1, kernel=3, up=False, down=True,
                             resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul, name='conv_1')
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_1')

        # resnet skip
        self.conv_skip = Conv2D(fmaps=self.n_f1, kernel=1, up=False, down=True,
                                resample_kernel=[1, 3, 3, 1], gain=self.gain, lrmul=self.lrmul, name='skip')

    def call(self, inputs, training=None, mask=None):
        x = inputs
        residual = x

        # conv0
        x = self.conv_0(x)
        x = self.apply_bias_act_0(x)

        # conv1 down
        x = self.conv_1(x)
        x = self.apply_bias_act_1(x)

        # resnet skip
        residual = self.conv_skip(residual)
        x = (x + residual) * self.resnet_scale
        return x

    # def get_config(self):
    #     config = super(DiscriminatorBlock, self).get_config()
    #     config.update({
    #         'n_f0': self.n_f0,
    #         'n_f1': self.n_f1,
    #         'gain': self.gain,
    #         'lrmul': self.lrmul,
    #         'res': self.res,
    #         'resnet_scale': self.resnet_scale,
    #     })
    #     return config


class DiscriminatorLastBlock(tf.keras.layers.Layer):
    def __init__(self, n_f0, n_f1, **kwargs):
        super(DiscriminatorLastBlock, self).__init__(**kwargs)
        self.gain = 1.0
        self.lrmul = 1.0
        self.n_f0 = n_f0
        self.n_f1 = n_f1

        self.minibatch_std = MinibatchStd(group_size=4, num_new_features=1, name='minibatchstd')

        # conv_0
        self.conv_0 = Conv2D(fmaps=self.n_f0, kernel=3, up=False, down=False,
                             resample_kernel=None, gain=self.gain, lrmul=self.lrmul, name='conv_0')
        self.apply_bias_act_0 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_0')

        # dense_1
        self.dense_1 = Dense(self.n_f1, gain=self.gain, lrmul=self.lrmul, name='dense_1')
        self.apply_bias_act_1 = BiasAct(lrmul=self.lrmul, act='lrelu', name='bias_1')

    def call(self, x, training=None, mask=None):
        x = self.minibatch_std(x)

        # conv_0
        x = self.conv_0(x)
        x = self.apply_bias_act_0(x)

        # dense_1
        x = self.dense_1(x)
        x = self.apply_bias_act_1(x)
        return x

    # def get_config(self):
    #     config = super(DiscriminatorLastBlock, self).get_config()
    #     config.update({
    #         'n_f0': self.n_f0,
    #         'n_f1': self.n_f1,
    #         'gain': self.gain,
    #         'lrmul': self.lrmul,
    #         'res': self.res,
    #     })
    #     return config