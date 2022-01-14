import numpy as np

from cuda.upfirdn_2d import *
from cuda.fused_bias_act import fused_bias_act

def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = tf.reduce_prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    fan_in = tf.cast(fan_in, dtype=tf.float32)
    he_std = gain / tf.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef

def lerp(a, b, t):
    out = a + (b - a) * t
    return out

def lerp_clip(a, b, t):
    out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out

##################################################################################
# Layers
##################################################################################

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, up, down, resample_kernel, gain, lrmul, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.up = up
        self.down = down

        self.k, self.pad0, self.pad1 = compute_paddings(resample_kernel, self.kernel, up, down, is_conv=True)

    def build(self, input_shape):
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        w = self.runtime_coef * self.w

        # actual conv
        if self.up:
            x = upsample_conv_2d(x, w, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        elif self.down:
            x = conv_downsample_2d(x, w, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        else:
            x = tf.nn.conv2d(x, w, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
        return x

    # def get_config(self):
    #     config = super(Conv2D, self).get_config()
    #     config.update({
    #         'in_res': self.in_res,
    #         'in_fmaps': self.in_fmaps,
    #         'fmaps': self.fmaps,
    #         'kernel': self.kernel,
    #         'gain': self.gain,
    #         'lrmul': self.lrmul,
    #         'up': self.up,
    #         'down': self.down,
    #         'k': self.k,
    #         'pad0': self.pad0,
    #         'pad1': self.pad1,
    #         'runtime_coef': self.runtime_coef,
    #     })
    #     return config

class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, style_fmaps, kernel, up, down, demodulate, resample_kernel, gain, lrmul, fused_modconv, **kwargs):
        super(ModulatedConv2D, self).__init__(**kwargs)
        assert not (up and down)

        self.fmaps = fmaps
        self.style_fmaps = style_fmaps
        self.kernel = kernel
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.fused_modconv = fused_modconv
        self.gain = gain
        self.lrmul = lrmul

        self.k, self.pad0, self.pad1 = compute_paddings(resample_kernel, self.kernel, up, down, is_conv=True)

        # self.factor = 2
        self.mod_dense = Dense(self.style_fmaps, gain=1.0, lrmul=1.0, name='mod_dense')
        self.mod_bias = BiasAct(lrmul=1.0, act='linear', name='mod_bias')

    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        in_fmaps = x_shape[1]
        weight_shape = [self.kernel, self.kernel, in_fmaps, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        # [kkIO]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def scale_conv_weights(self, w):
        # convolution kernel weights for fused conv
        weight = self.runtime_coef * self.w  # [kkIO]
        weight = weight[np.newaxis]  # [BkkIO]

        # modulation
        style = self.mod_dense(w)  # [BI]
        style = self.mod_bias(style) + 1.0  # [BI]
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]  # [BkkIO]

        # demodulation
        d = None
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)  # [BO]
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO]

        return weight, style, d

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        # height, width = tf.shape(x)[2], tf.shape(x)[3]

        # prepare weights: [BkkIO] Introduce minibatch dimension
        # prepare convoultuon kernel weights
        weight, style, d = self.scale_conv_weights(y)

        if self.fused_modconv:
            # Fused => reshape minibatch to convolution groups
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])

            # weight: reshape, prepare for fused operation
            new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]  # [kkI(BO)]
            weight = tf.transpose(weight, [1, 2, 3, 0, 4])  # [kkIBO]
            weight = tf.reshape(weight, shape=new_weight_shape)  # [kkI(BO)]
        else:
            # [BIhw] Not fused => scale input activations
            x *= style[:, :, tf.newaxis, tf.newaxis]

        # Convolution with optional up/downsampling.
        if self.up:
            x = upsample_conv_2d(x, weight, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        elif self.down:
            x = conv_downsample_2d(x, weight, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # Reshape/scale output
        if self.fused_modconv:
            # Fused => reshape convolution groups back to minibatch
            x_shape = tf.shape(x)
            x = tf.reshape(x, [-1, self.fmaps, x_shape[2], x_shape[3]])
        elif self.demodulate:
            # [BOhw] Not fused => scale output activations
            x *= d[:, :, tf.newaxis, tf.newaxis]

        return x

    # def get_config(self):
    #     config = super(ModulatedConv2D, self).get_config()
    #     config.update({
    #         'in_res': self.in_res,
    #         'in_fmaps': self.in_fmaps,
    #         'fmaps': self.fmaps,
    #         'kernel': self.kernel,
    #         'demodulate': self.demodulate,
    #         'fused_modconv': self.fused_modconv,
    #         'gain': self.gain,
    #         'lrmul': self.lrmul,
    #         'up': self.up,
    #         'down': self.down,
    #         'k': self.k,
    #         'pad0': self.pad0,
    #         'pad1': self.pad1,
    #         'runtime_coef': self.runtime_coef,
    #     })
    #     return config

class Dense(tf.keras.layers.Layer):
    def __init__(self, fmaps, gain, lrmul, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        fan_in = tf.reduce_prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        weight = self.runtime_coef * self.w

        c = tf.reduce_prod(tf.shape(inputs)[1:])
        x = tf.reshape(inputs, shape=[-1, c])
        x = tf.matmul(x, weight)
        return x

    # def get_config(self):
    #     config = super(Dense, self).get_config()
    #     config.update({
    #         'fmaps': self.fmaps,
    #         'gain': self.gain,
    #         'lrmul': self.lrmul,
    #         'runtime_coef': self.runtime_coef,
    #     })
    #     return config

class LabelEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(LabelEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        weight_shape = [input_shape[1], self.embed_dim]
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.matmul(inputs, self.w)
        return x

    # def get_config(self):
    #     config = super(LabelEmbedding, self).get_config()
    #     config.update({
    #         'embed_dim': self.embed_dim,
    #     })
    #     return config

##################################################################################
# Blocks
##################################################################################
class FromRGB(tf.keras.layers.Layer):
    def __init__(self, fmaps, **kwargs):
        super(FromRGB, self).__init__(**kwargs)
        self.fmaps = fmaps

        self.conv = Conv2D(fmaps=self.fmaps, kernel=1, up=False, down=False,
                           resample_kernel=None, gain=1.0, lrmul=1.0, name='conv')
        self.apply_bias_act = BiasAct(lrmul=1.0, act='lrelu', name='bias')

    def call(self, inputs, training=None, mask=None):
        y = self.conv(inputs)
        y = self.apply_bias_act(y)
        return y

    # def get_config(self):
    #     config = super(FromRGB, self).get_config()
    #     config.update({
    #         'fmaps': self.fmaps,
    #         'res': self.res,
    #     })
    #     return config

class ToRGB(tf.keras.layers.Layer):
    def __init__(self, in_ch, **kwargs):
        super(ToRGB, self).__init__(**kwargs)
        self.in_ch = in_ch

        self.conv = ModulatedConv2D(fmaps=3, style_fmaps=in_ch, kernel=1, up=False, down=False, demodulate=False,
                                    resample_kernel=None, gain=1.0, lrmul=1.0, fused_modconv=True, name='conv')
        self.apply_bias = BiasAct(lrmul=1.0, act='linear', name='bias')

    def call(self, inputs, training=None, mask=None):
        x, w = inputs

        x = self.conv([x, w])
        x = self.apply_bias(x)
        return x

    # def get_config(self):
    #     config = super(ToRGB, self).get_config()
    #     config.update({
    #         'in_ch': self.in_ch,
    #         'res': self.res,
    #     })
    #     return config

class PE2d(tf.keras.layers.Layer):
    def __init__(self, channel, height, width, scale=1.0):
        super(PE2d, self).__init__()
        if channel % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(channel))

        height = int(height * scale)
        width = int(width * scale)
        self.pe = np.zeros(shape=[channel, height, width], dtype=np.float32)

        # Each dimension use half of d_model
        self.d_model = int(channel / 2)
        self.div_term = np.exp(np.arange(0., self.d_model, 2.) * -(np.log(10000.) / self.d_model)) / scale
        self.pos_h = np.expand_dims(np.arange(0., height), axis=-1) # [4, 1]
        self.pos_w = np.expand_dims(np.arange(0., width), axis=-1)


        self.gamma = tf.Variable(initial_value=tf.ones(shape=[1], dtype=tf.float32), trainable=True)

    def call(self, inputs, shift_h=0, shift_w=0, training=None, mask=None):
        pos_h = np.roll(self.pos_h, round(shift_h), 0) + (round(shift_h) - shift_h)
        pos_w = np.roll(self.pos_w, round(shift_w), 0) + (round(shift_w) - shift_w)

        self.pe[0:self.d_model:2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.sin(pos_w * self.div_term),
                    axes=[1, 0]),
                axis=1),
            reps=[1, pos_h.shape[0], 1])

        self.pe[1:self.d_model:2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.cos(pos_w * self.div_term),
                    axes=[1, 0]),
                axis=1),
            reps=[1, pos_h.shape[0], 1])

        self.pe[self.d_model::2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.sin(pos_h * self.div_term),
                    axes=[1, 0]),
                axis=2),
            reps=[1, 1, pos_w.shape[0]])

        self.pe[self.d_model + 1::2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.cos(pos_h * self.div_term),
                    axes=[1, 0]),
                axis=2),
            reps=[1, 1, pos_w.shape[0]])

        x = tf.cast(inputs, dtype=tf.float32) + self.gamma * np.expand_dims(self.pe, axis=0)

        return x

class PE2dStart(tf.keras.layers.Layer):
    def __init__(self, channel, height, width, scale=1.0):
        super(PE2dStart, self).__init__()
        if channel % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(channel))

        height = int(height * scale)
        width = int(width * scale)
        self.pe = np.zeros(shape=[channel, height, width])

        # Each dimension use half of d_model
        self.d_model = int(channel / 2)
        self.div_term = np.exp(np.arange(0., self.d_model, 2.) * -(np.log(10000.) / self.d_model)) / scale
        self.pos_h = np.expand_dims(np.arange(0., height), axis=-1) # [4, 1]
        self.pos_w = np.expand_dims(np.arange(0., width), axis=-1)

    def call(self, inputs, shift_h=0, shift_w=0, training=None, mask=None):
        pos_h = np.roll(self.pos_h, round(shift_h), 0) + (round(shift_h) - shift_h)
        pos_w = np.roll(self.pos_w, round(shift_w), 0) + (round(shift_w) - shift_w)

        self.pe[0:self.d_model:2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.sin(pos_w * self.div_term),
                    axes=[1, 0]),
                axis=1),
            reps=[1, pos_h.shape[0], 1])

        self.pe[1:self.d_model:2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.cos(pos_w * self.div_term),
                    axes=[1, 0]),
                axis=1),
            reps=[1, pos_h.shape[0], 1])

        self.pe[self.d_model::2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.sin(pos_h * self.div_term),
                    axes=[1, 0]),
                axis=2),
            reps=[1, 1, pos_w.shape[0]])

        self.pe[self.d_model + 1::2, :, :] = np.tile(
            np.expand_dims(
                np.transpose(
                    np.cos(pos_h * self.div_term),
                    axes=[1, 0]),
                axis=2),
            reps=[1, 1, pos_w.shape[0]])

        x = np.tile(np.expand_dims(self.pe, axis=0), reps=[inputs.shape[0], 1, 1, 1])

        return x

class ConstantInput(tf.keras.layers.Layer):
    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()

        const_init = tf.random.normal(shape=(1, channel, size, size), mean=0.0, stddev=1.0)
        self.const = tf.Variable(const_init, name='const', trainable=True)

    def call(self, inputs, training=None, mask=None):
        batch = inputs.shape[0]
        x = tf.tile(self.const, multiples=[batch, 1, 1, 1])

        return x

##################################################################################
# etc
##################################################################################
class BiasAct(tf.keras.layers.Layer):
    def __init__(self, lrmul, act, **kwargs):
        super(BiasAct, self).__init__(**kwargs)
        self.lrmul = lrmul
        self.act = act

    def build(self, input_shape):
        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b
        x = fused_bias_act(inputs, b=b, act=self.act, alpha=None, gain=None)
        return x

    # def get_config(self):
    #     config = super(BiasAct, self).get_config()
    #     config.update({
    #         'lrmul': self.lrmul,
    #         'act': self.act,
    #     })
    #     return config

class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=True, name='w')


    def call(self, inputs, noise=None, training=None, mask=None):
        x_shape = tf.shape(inputs)

        # noise: [1, 1, x_shape[2], x_shape[3]] or None
        if noise is None:
            noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.float32)

        x = inputs + noise * self.noise_strength
        return x

    def get_config(self):
        config = super(Noise, self).get_config()
        config.update({})
        return config

class MinibatchStd(tf.keras.layers.Layer):
    def __init__(self, group_size, num_new_features, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, inputs, training=None, mask=None):
        s = tf.shape(inputs)
        group_size = tf.minimum(self.group_size, s[0])

        y = tf.reshape(inputs, [group_size, -1, self.num_new_features, s[1] // self.num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=[2])
        y = tf.cast(y, inputs.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3]])

        x = tf.concat([inputs, y], axis=1)
        return x

    def get_config(self):
        config = super(MinibatchStd, self).get_config()
        config.update({
            'group_size': self.group_size,
            'num_new_features': self.num_new_features,
        })
        return config

def torch_normalization(x):
    x /= 255.

    r, g, b = tf.split(axis=-1, num_or_size_splits=3, value=x)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x = tf.concat(axis=-1, values=[
        (r - mean[0]) / std[0],
        (g - mean[1]) / std[1],
        (b - mean[2]) / std[2]
    ])

    return x


def inception_processing(filename):
    x = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
    img = tf.image.resize(img, [256, 256], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
    img = tf.image.resize(img, [299, 299], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

    img = torch_normalization(img)
    # img = tf.transpose(img, [2, 0, 1])
    return img