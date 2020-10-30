import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm

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

class frechetInceptionDistance(object):
    def __init__(self,mu1=[],sigma1=[]):
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
        self.mu1 = np.array(mu1)
        self.sigma1 = np.array(sigma1)
    def get_real_img_stats(self,images1):
#         self.images1 = images1.astype('float32')
        images1 = preprocess_input(images1)
        print('preprocess complete, new shape:',images1.shape)
        act1 = self.model.predict(images1)
        print('prediction complete, act1 shape:',act1.shape)
        self.mu1, self.sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        print('mu1 sigma1 shape:',self.mu1.shape,self.sigma1.shape)
        return self.mu1, self.sigma1
    def calculate_fid(self, images2=None, do_prepocess = False):
#         images2 = images2.astype('float32')
        if do_prepocess:
            images2 = preprocess_input(images2)        
        act2 = self.model.predict(images2)
        # calculate mean and covariance statistics        
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((self.mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(self.sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(self.sigma1 + sigma2 - 2.0 * covmean)
        return fid
    def scale_images(self, images=None, new_shape=None):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return asarray(images_list)

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