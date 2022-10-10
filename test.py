import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from metrics1 import dice_coef,precision, recall
from loss import dice_coef_loss
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

import warnings
warnings.simplefilter('ignore')

H=256
W=256

def load_data(dataset_path1, split=0.2):
    
    test_x = glob(os.path.join(dataset_path1, "*.jpg")) 
    test_x.sort()
    
    
    return test_x


def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x                                   #original resized image
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)


def read_mask(path):
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    y = cv2.resize(y, (W, H))
    ori_y = y
    y = y/255.0
    y = y > 0.5
    y = y.astype(np.int32)                    ## (256, 256)
    return ori_y, y


def save_results(ori_x, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    y_pred = np.expand_dims(y_pred, axis=-1)*255.0  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)*255.0 ## (256, 256, 3)
    
    
    cat_images = np.concatenate([ori_x, line, y_pred], axis=1)
    #print(cat_images.shape)
    #print(ori_x.shape)
    #print(ori_y.shape)
    #print(y_pred.shape)
    #print(save_image_path)
    #cv2.imwrite(save_image_path, cat_images)
    
    cv2.imwrite(save_image_path, cat_images)
    
def get_id_from_file_path(file_path, indicator):
    return file_path.split(os.path.sep)[-1].replace(indicator, '')

if __name__ == "__main__":
    
    #Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    
    #Dataset 
    path1="../Datasets/DFUC2022/DFUC2022_test_release/"

    batch_size = 8
    test_x = load_data(path1)
    

    print(len(test_x))
    print(test_x[1])

    """ Loading model """
    with CustomObjectScope({'precision': precision, 'recall':recall, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("files/model4.h5")

    
    #model = tf.keras.models.load_model("files/model3.h5", custom_objects={'precision': precision, 'recall':recall, 'dice_coef': dice_coef})
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0), metrics=["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), dice_coef])

    #model.evaluate(test_x, steps=50)
    
    
    SCORE = []

for x in tqdm(test_x):

    """Exctracting the image name """
    name = get_id_from_file_path(x, ".jpg")
        
    """ Read the image and mask """
    ori_x, x = read_image(x)

    """ Predicting the mask """
    y_pred = model.predict(x)[0] > 0.5          #threshold=0.5
    y_pred = np.squeeze(y_pred, axis=-1)        #-1= last axis
    y_pred = y_pred.astype(np.int8)

    """ Saving the predicted mask """
    save_image_path = f"augtest/{name}.png"
    save_results(ori_x, y_pred, save_image_path)
