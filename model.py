from pdb import set_trace as TT
import tensorflow as tf
from gym import spaces
import numpy as np
from stable_baselines.common.policies import ActorCriticPolicy, FeedForwardPolicy
from stable_baselines.common.distributions import CategoricalProbabilityDistributionType, ProbabilityDistributionType, CategoricalProbabilityDistribution, ProbabilityDistribution, DiagGaussianProbabilityDistributionType, MultiCategoricalProbabilityDistributionType
from stable_baselines.a2c.utils import conv, linear, conv_to_fc

def NCA(x, channel_n, n_tools, angle=0.0, step_size=1.0,  **kwargs):
    relu = tf.nn.relu
    sigmoid = tf.nn.sigmoid
#   pre_life_mask = get_living_mask(x)
#   y = perceive(x, channel_n, angle=angle)
    y = relu(conv(x, 'c1', n_filters=512, filter_size=3, stride=1, pad='SAME', init_scale=np.sqrt(2)))
    dx = conv(y, 'c2', n_filters=channel_n, filter_size=1, stride=1, pad='SAME', init_scale=np.sqrt(2))
    dx = dx * step_size
    fire_rate = 0.01
    if fire_rate is None:
        fire_rate = fire_rate
#   x += dx
#   x = relu(x)
    update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
    x += dx * tf.cast(update_mask, tf.float32)
#   post_life_mask = get_living_mask(x)
#   life_mask = pre_life_mask & post_life_mask
#   x = x * tf.cast(life_mask, tf.float32)
    x = sigmoid(conv(x, 'c3', n_filters=n_tools, filter_size=1, stride=1, pad='SAME', init_scale=np.sqrt(2)))
    val = conv_to_fc(x)
    val = sigmoid(linear(val, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
    act = conv_to_fc(x)
    return act, val

#   val = relu(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
#                    init_scale=np.sqrt(2)))
#   val = relu(conv(val, 'v2', n_filters=64, filter_size=3, stride=2,
#                    init_scale=np.sqrt(3)))
#   # val = activ(conv(val, 'v3', n_filters=64, filter_size=3, stride=2,
#   #    init_scale=np.sqrt(2)))
#   val = relu(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
#                    init_scale=np.sqrt(2)))
#   val = conv_to_fc(val)

def perceive(x, channel_n, angle=0.0):
    identify = np.float32([0, 1, 0])
    identify = np.outer(identify, identify)
    dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
    dy = dx.T
    c, s = tf.cos(angle), tf.sin(angle)
    kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
    kernel = tf.repeat(kernel, channel_n, 2)
    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
    return y


def get_living_mask(x):
    alpha = x[:, :, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') > 0.1

#class CAModel(tf.keras.Model):
#
#  def __init__(self, channel_n, n_tools, **kwargs):
#      from tensorflow.keras import layers
#      from tensorflow.keras.layers import Conv2d
#
#      super().__init__()
#      self.channel_n = channel_n
#      self.fire_rate = 0.1
#
#      self.dmodel = tf.keras.Sequential([
#            Conv2D(128, 1, activation=tf.nn.relu),
#            Conv2D(self.channel_n, 1, activation=None,
#                kernel_initializer=tf.zeros_initializer),
#      ])
#
#      self(tf.zeros([1, 3, 3, channel_n]))  # dummy call to build the model
#
#  @tf.function
#  def perceive(self, x, angle=0.0):
#    identify = np.float32([0, 1, 0])
#    identify = np.outer(identify, identify)
#    dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
#    dy = dx.T
#    c, s = tf.cos(angle), tf.sin(angle)
#    kernel = tf.stack([identify, c*dx-s*dy, s*dx+c*dy], -1)[:, :, None, :]
#    kernel = tf.repeat(kernel, self.channel_n, 2)
#    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
#    return y
#
#  @tf.function
#  def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
#    pre_life_mask = get_living_mask(x)
#
#    y = self.perceive(x, angle)
#    dx = self.dmodel(y)*step_size
#    if fire_rate is None:
#      fire_rate = self.fire_rate
#    update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate
#    x += dx * tf.cast(update_mask, tf.float32)
#
#    post_life_mask = get_living_mask(x)
#    life_mask = pre_life_mask & post_life_mask
#    return x * tf.cast(life_mask, tf.float32)



def Cnn1(image, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

def Cnn2(image, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

def FullyConv1(image, n_tools, **kwargs):
    activ = tf.nn.relu
    x = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c2', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c3', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c4', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c5', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c6', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c7', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c8', n_filters=n_tools, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    act = conv_to_fc(x)
    val = activ(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
        init_scale=np.sqrt(2)))
    val = conv_to_fc(val)
    return act, val


def FullyConv2(image, n_tools, **kwargs):
    activ = tf.nn.relu
    x = activ(conv(image, 'c1', n_filters=32, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c2', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c3', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c4', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c5', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c6', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c7', n_filters=64, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    x = activ(conv(x, 'c8', n_filters=n_tools, filter_size=3, stride=1,
        pad='SAME', init_scale=np.sqrt(2)))
    act = conv_to_fc(x)
    val = activ(conv(x, 'v1', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v2', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(3)))
   #val = activ(conv(val, 'v3', n_filters=64, filter_size=3, stride=2,
   #    init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
        init_scale=np.sqrt(2)))
    val = conv_to_fc(val)
    return act, val


def ValShrink(val, **kwargs):
    activ = tf.nn.relu
    val = activ(conv(val, 'v1', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v2', n_filters=64, filter_size=3, stride=2,
        init_scale=np.sqrt(3)))
   #val = activ(conv(val, 'v3', n_filters=64, filter_size=3, stride=2,
   #    init_scale=np.sqrt(2)))
    val = activ(conv(val, 'v4', n_filters=64, filter_size=1, stride=1,
        init_scale=np.sqrt(2)))
    val = conv_to_fc(val)
    return val


def FractalNet(image, n_tools, n_recs, blocks=[64], **kwargs):
    '''
     - blocks: a list, ordered from network in to out, of each block's n_chan
    '''
    x = layers.Conv2D(blocks[0], 1, 1, activation='relu')(image) # embedding
    for n_chan in blocks:
        x = FractalBlock(x, n_recs, n_chan, **kwargs)
    act = layers.Conv2D(n_tools, 1, 1, activation='relu')(x)
    act = conv_to_fc(act)
    val = layers.Conv2D(1, 1, 1, activation='relu')(x)
    val = conv_to_fc(val)
    return act, val


def FractalBlock(image, n_recs, n_chan, **kwargs):
    x = layers.Conv2D(n_chan, 1, 1, activation='relu')(image) # embed
    child = None
    for i in range(n_recs):
        child = SubFractal(child, n_chan, **kwargs)
    x = tf.expand_dims(x, 0)
    x = child(x)
    x = tf.squeeze(x, 0)
    return x


class SubFractal(tf.Module):
    def __init__(self, child, n_chan, **kwargs):
        '''
            -child: a SubFractal or None, if base case
        '''
        self.child = child
        self.skip = AtomicNode(n_chan)


    def __call__(self, x, join=True):
        '''
        - join: is this subfractal responsible for joining the accumulated outputs?
        '''
        x = self.skip(x)
        if self.child:
            x_body = self.child(self.child(x, join=True), join=False)
            x = tf.concat((x, x_body), 0)
        if join:
            x = tf.math.reduce_mean(x, 0, keepdims=True)
        return x


class AtomicNode(tf.Module):
    def __init__(self, n_chan):
        self.c1 = layers.Conv2D(n_chan, 3, 1, padding='same', activation='relu')

    def __call__(self, x):
        print(x.shape)
        x = tf.squeeze(x, 0)
        x = self.c1(x)
        x = tf.expand_dims(x, 0)
        return x


class NoDenseCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_cat):
        """
        The probability distribution type for categorical input

        :param n_cat: (int) the number of categories
        """
        self.n_cat = n_cat

    def probability_distribution_class(self):
        return CategoricalProbabilityDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0,
                                       init_bias=0.0):
        pdparam = pi_latent_vector
        q_values = vf_latent_vector
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [self.n_cat]

    def sample_shape(self):
        return []

    def sample_dtype(self):
        return tf.int64


class NoDenseMultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
    def __init__(self, n_vec):
        """
        The probability distribution type for multiple categorical input

        :param n_vec: ([int]) the vectors
        """
        # Cast the variable because tf does not allow uint32
        self.n_vec = n_vec.astype(np.int32)
        # Check that the cast was valid
        assert (self.n_vec > 0).all(), "Casting uint32 to int32 was invalid"

    def probability_distribution_class(self):
        return NoDenseMultiCategoricalProbabilityDistribution

    def proba_distribution_from_flat(self, flat):
        return NoDenseMultiCategoricalProbabilityDistribution(self.n_vec, flat)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        # pdparam = linear(pi_latent_vector, 'pi', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        pdparam = pi_latent_vector
        q_values = linear(vf_latent_vector, 'q', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), pdparam, q_values

    def param_shape(self):
        return [sum(self.n_vec)]

    def sample_shape(self):
        return [len(self.n_vec)]

    def sample_dtype(self):
        return tf.int64


class NoDenseMultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(map(CategoricalProbabilityDistribution, tf.split(flat, nvec, axis=-1)))
        super(NoDenseMultiCategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.stack([p.mode() for p in self.categoricals], axis=-1)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.stack([p.sample() for p in self.categoricals], axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError


class FullyConvPolicyBigMap(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super(FullyConvPolicyBigMap, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs)
        n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
        self._pdtype = NoDenseCategoricalProbabilityDistributionType(ac_space.n)
        with tf.variable_scope("model", reuse=kwargs['reuse']):
            pi_latent, vf_latent = FullyConv2(self.processed_obs, n_tools, **kwargs)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class CAPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super(CAPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs)
        n_tools = ac_space.nvec[0]
        channel_n = ob_space.shape[2]
        # n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
        # self._pdtype = DiagGaussianProbabilityDistributionType(ac_space.n)
        self._pdtype = NoDenseMultiCategoricalProbabilityDistributionType(ac_space.nvec)
        with tf.variable_scope("model", reuse=kwargs['reuse']):
            pi_latent, vf_latent = NCA(self.processed_obs, channel_n=channel_n, n_tools=n_tools, **kwargs)
#           pi_latent, vf_latent = CAModel(self.processed_obs, channel_n=channel_n, n_tools=n_tools, **kwargs)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
#       if deterministic:
        if True:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


#TODO: SCRAP THIS?
class FullyConvPolicySmallMap(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super(FullyConvPolicySmallMap, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs)
        n_tools = int(ac_space.n / (ob_space.shape[0] * ob_space.shape[1]))
        self._pdtype = NoDenseCategoricalProbabilityDistributionType(ac_space.n)
        with tf.variable_scope("model", reuse=kwargs['reuse']):
            pi_latent, vf_latent = FullyConv1(self.processed_obs, n_tools, **kwargs)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class CustomPolicyBigMap(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicyBigMap, self).__init__(*args, **kwargs, cnn_extractor=Cnn2, feature_extraction="cnn")

class CustomPolicySmallMap(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicySmallMap, self).__init__(*args, **kwargs, cnn_extractor=Cnn1, feature_extraction="cnn")
