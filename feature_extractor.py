if True:
    from utils import reset_random

    reset_random()

import glob
import h5py
import tqdm
import os
import numpy as np
from utils import CLASSES

SHAPE = (224, 224, 3)


def resnet_152():
    from tensorflow.keras.applications.resnet import ResNet152
    from tensorflow.keras import Input
    model = ResNet152(include_top=False,
                      weights='weights/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5',
                      input_tensor=Input(shape=SHAPE))
    return model


def extract_feature(img_p, model):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    img = image.load_img(img_p, target_size=SHAPE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    feat = feature.flatten()
    return feat


if __name__ == '__main__':
    data_path = 'Data/segmented'
    data_save_path = 'Data/features'
    mdl = resnet_152()
    os.makedirs(data_save_path, exist_ok=True)
    features = []
    labels = []
    for cls in CLASSES:
        for img_path in tqdm.tqdm(glob.glob(os.path.join(data_path, cls, '*')),
                                  desc='Extracting Features For Class :: {0}'.format(cls)):
            features.append(extract_feature(img_path, mdl))
            labels.append(CLASSES.index(cls))

    h5f_data = h5py.File(os.path.join(data_save_path, 'features.h5'), 'w')
    h5f_data.create_dataset('features', data=np.array(features))
    h5f_data.close()

    h5f_data = h5py.File(os.path.join(data_save_path, 'labels.h5'), 'w')
    h5f_data.create_dataset('labels', data=np.array(labels))
    h5f_data.close()
