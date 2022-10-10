import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import Xnet
from metrics1 import dice_coef,precision, recall
from loss import dice_coef_loss
from tensorflow.keras.losses import binary_crossentropy
from matplotlib import pyplot
from tensorflow import keras
import datetime
import json

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(dataset_path1,dataset_path2, split=0.2, random_state=42):
    images = glob(os.path.join(dataset_path1, "*.jpg"))
    images.sort()
    masks = glob(os.path.join(dataset_path2, "*.png"))
    masks.sort()
    
    #images, masks = shuffling(images, masks)
    
    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3) to read as rgb
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x                                ## (256, 256, 3) returns image array

def read_mask(path):
    path = path.decode()
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W) to read img as grayscale
    y = cv2.resize(y, (W, H))
    y = y/255.0
    y = y.astype(np.float32)                    ## (256, 256)
    y = np.expand_dims(y, axis=-1)              ## (256, 256, 1) here we added channel
    return y

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y                                #return numpy array ,returns float 32

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch):                       #X= list of images Y= list of masks
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)                 #to read the path in tensors
    dataset = dataset.batch(batch)                  #provide batch_size
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    
    print("Training Unet ++")
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Folder for saving data """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 4
    num_epoch =50
    loss = 'binary_crossentropy'
    lr=1e-4
    
    """create path to save weights"""
    model_path = "files/model4.h5"

    """create path to save all metrics"""
    csv_path = "files/data4.csv"

    """ Dataset : 60/20/20 """
    dataset_path1="../UnetPlusPlus_DFUCC/new_data/DFUC2022_train_images/"
    dataset_path2="../UnetPlusPlus_DFUCC/new_data/DFUC2022_train_masks/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path1,dataset_path2)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    


    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch_size)

    
    """ Model """
    
    model = Xnet(backbone_name="efficientnetb7", encoder_weights="imagenet", classes=1, activation="sigmoid")    
    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    metrics = [dice_coef, precision, recall]
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    #model.summary()

    #will call callbacks at training
    #will call callbacks at training
    callbacks = [
        
        ModelCheckpoint(model_path, verbose=1, save_best_only=False ),            #to save weightfile while training
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),                                                    #to log all the data while training
        TensorBoard(),                                                          #for visualization
        EarlyStopping(monitor='loss', patience=100, mode='max',verbose=1, restore_best_weights=True)
      ]

    train_steps = len(train_x)//batch_size
    valid_steps = len(valid_x)//batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    history = model.fit(train_dataset,
        validation_data=valid_dataset,
        shuffle=True,
        epochs=num_epoch,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks)
    
    # plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.title('Model Loss')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Validation'], loc='upper right')

    pyplot.savefig("../UnetPlusPlus_DFUCC/graph/augtrain.png")
    pyplot.legend()
    pyplot.show()
    