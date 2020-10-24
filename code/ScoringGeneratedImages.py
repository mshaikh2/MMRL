import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
# import tensorflow as tf
from skimage.transform import resize
from numpy import asarray
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

class inceptionScore(object):
    def __init__(self):
        self.model = InceptionV3()
        
    def calculate_inception_score(self, images=None, do_prepocess = False, n_split=10, eps=1e-16):
        # load inception v3 model

    #     with tf.device('/GPU:1'):
    #     model = InceptionV3()
        # enumerate splits of images/predictions
        scores = list()
        n_part = floor(images.shape[0] / n_split)
        for i in range(n_split):
            # retrieve images
            ix_start, ix_end = i * n_part, (i+1) * n_part
            subset = images[ix_start:ix_end]
            # convert from uint8 to float32
            subset = subset.astype('float32')
            # scale images to the required size
            subset = self.scale_images(subset, (299,299,3))
            # pre-process images, scale to [-1,1]
            if do_prepocess:
                subset = preprocess_input(subset)
    #         print(subset.min(),subset.max())
            # predict p(y|x)
    #         with tf.device('/GPU:1'):
#             print('prediction started ...')
            p_yx = self.model.predict(subset)
            # calculate p(y)
            p_y = expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = mean(sum_kl_d)
            # undo the log
            is_score = exp(avg_kl_d)
            # store
            scores.append(is_score)
        # average across images
        is_avg, is_std = mean(scores), std(scores)
#         print('is_avg, is_std:',is_avg, is_std)
        return is_avg, is_std

# scale an array of images to a new size
    def scale_images(self, images=None, new_shape=None):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return asarray(images_list)