if True:
    from utils import reset_random
    reset_random()

import glob
import os
import tqdm
import numpy as np
from keras.models import load_model
from keras import backend as keras
import cv2
from utils import CLASSES


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def segment_image(image_path, model):
    orig_image = cv2.imread(image_path)
    image = cv2.resize(orig_image, (512, 512))[:, :, 0]
    image = (image.reshape((1, 512, 512, 1)) - 127.0) / 127.0
    pred = model.predict([image]).reshape((512, 512))
    return cv2.resize(cv2.imread(image_path, 0), (512, 512)), np.array(pred, dtype=np.uint8)


def get_segmented_data(image, mask):
    mask[np.where((mask == 1))] = 255
    segmented = cv2.bitwise_and(image.copy(), image.copy(), mask=mask.copy())
    image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
    mask_rgb[np.where((mask_rgb == [255, 255, 255]).all(axis=2))] = [255, 0, 0]
    mask_image = cv2.bitwise_or(image_rgb.copy(), mask_rgb.copy())
    drawn = cv2.addWeighted(mask_image.copy(), 1, image_rgb.copy(), 0, 1)
    return mask_rgb, segmented, drawn


if __name__ == '__main__':
    data_path = 'Data/data'
    data_save_path1 = 'Data/mask'
    data_save_path2 = 'Data/segmented'
    data_save_path3 = 'Data/mask_drawn'
    segmentation_model = load_model('unet-model/model.h5',
                                    custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    for cls in CLASSES:
        images_list = sorted(glob.glob(os.path.join(data_path, cls, '*')))
        data_save_path1_ = os.path.join(data_save_path1, cls)
        data_save_path2_ = os.path.join(data_save_path2, cls)
        data_save_path3_ = os.path.join(data_save_path3, cls)
        os.makedirs(data_save_path1_, exist_ok=True)
        os.makedirs(data_save_path2_, exist_ok=True)
        os.makedirs(data_save_path3_, exist_ok=True)
        for img_path in tqdm.tqdm(images_list, desc='[INFO] Segmenting X-RAY Images For Class :: {0}'.format(cls)):
            image_, mask_ = segment_image(img_path, segmentation_model)
            mask_, segmented_, drawn_ = get_segmented_data(image_, mask_)
            cv2.imwrite(os.path.join(data_save_path1_, os.path.basename(img_path)), mask_)
            cv2.imwrite(os.path.join(data_save_path2_, os.path.basename(img_path)), segmented_)
            cv2.imwrite(os.path.join(data_save_path3_, os.path.basename(img_path)), drawn_)
