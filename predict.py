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
from data import tf_dataset
from metrics1 import dice_coef,precision, recall
from loss import dice_coef_loss
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

H=256
W=256

def load_data(dataset_path1, dataset_path2, split=0.2):
    train_images = glob(os.path.join(dataset_path1, "*.jpg"))
    train_images.sort()
    train_masks = glob(os.path.join(dataset_path2, "*.png"))
    train_images.sort()
    
    test_x = glob(os.path.join(dataset_path1, "*.jpg")) 
    test_x.sort()
    test_y = glob(os.path.join(dataset_path2, "*.png"))
    test_y.sort()
    
    train_x, valid_x, train_y, valid_y = train_test_split(train_images, train_masks, test_size=split,random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


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

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)*255.0 ## (256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    #print(cat_images.shape)
    #print(ori_x.shape)
    #print(ori_y.shape)
    #print(y_pred.shape)
    #print(save_image_path)
    cv2.imwrite(save_image_path, cat_images)


if __name__ == "__main__":
    
    #Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    
    #Dataset 
    path1="../Datasets/DFUC2022/DFUC2022_train_release/DFUC2022_train_images/"
    path2="../Datasets/DFUC2022/DFUC2022_train_release/DFUC2022_train_masks/"
    batch_size = 8
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path1,path2)
    
    print(len(train_x),len(valid_x),len(test_x))

    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)
    
    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

       
    #model = tf.keras.models.load_model("files/model.h5", custom_objects={'iou': iou, 'dice_coef': dice_coef})
    #model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0), metrics=["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou, dice_coef])

    # with CustomObjectScope({'iou': iou}):
    
    
    #model = tf.keras.models.load_model("files/model3.h5", custom_objects={'precision': precision, 'dice_coef': dice_coef, 'recall':recall})
    #model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0))"""

    model = tf.keras.models.load_model("files/model4.h5", custom_objects={'precision': precision, 'recall':recall, 'dice_coef': dice_coef})
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(0), metrics=["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), dice_coef])

    model.evaluate(test_dataset, steps=50)
    
    
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):               #tqdm=progress bar
        """Exctracting the image name """
        name = os.path.basename(x)
        
        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        """ Predicting the mask """
        y_pred = model.predict(x)[0] > 0.5          #threshold=0.5
        y_pred = np.squeeze(y_pred, axis=-1)        #-1= last axis
        y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask """
        save_image_path = f"augtrain/{name}"
        save_results(ori_x, ori_y, y_pred, save_image_path)

        """ Flatten the array """
        y = y.flatten()
        y_pred = y_pred.flatten()

        """ Calculating metrics values """
        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")     #zero_division=0
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")       #zero_division=0
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ mean metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns = ["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("../UnetPlusPlus_DFUCC/files/score.csv")

    
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('Model Loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper right')

    pyplot.savefig("../UnetPlusPlus_DFUCC/graph/loss.png")
    pyplot.legend()
    pyplot.show()
