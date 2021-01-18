# Module with functions to work with data.
import os
import numpy as np
import cv2
from tqdm import tqdm



def preprocess_image(image, image_size=64):
    """
    Description
    -----------
    A function processes single image.


    Parameters
    ----------
    image : str
        Image to process in array format, with shape (height, width)
    image_size : int
        Size to which resize image (on which model was trained).


    Returns
    -------
    resized: numpy.ndarray
        Processed image in array format with shape: (image_size, image_size), values in range: [0, 1].
   """

    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    idx = 0
    ls_xmin = []
    ls_ymin = []
    ls_xmax = []
    ls_ymax = []

    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        ls_xmin.append(x)
        ls_ymin.append(y)
        ls_xmax.append(x + w)
        ls_ymax.append(y + h)

    xmin = min(ls_xmin)
    ymin = min(ls_ymin)
    xmax = max(ls_xmax)
    ymax = max(ls_ymax)
    roi = image[ymin:ymax, xmin:xmax]
    resized = cv2.resize(roi, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resized = np.array(resized) / 255

    return resized


def load_image(path, image_size=64):
    """
    Description
    -----------
    A function that loads an processes single image.

    Parameters
    ----------
    path : str
      Path to image to load and process
    image_size : int
      Size to which resize image (on which model was trained).

    Returns
    -------
    image: numpy.ndarray
        Processed image in array format with shape: (image_size, image_size), values in range: [0, 1].
    """

    image = cv2.imread(path, 0)  # values: [0, 255] # shape: (height, width)
    image = preprocess_image(image, image_size)  # values:  [0, 1], shape: (image_size, image_size)

    return image


def load_images(path, image_size=64):
    """
    Description
    -----------
    A function that loads an processes bath of images.

    Parameters
    ----------
    path : str
        Path to dir with images to load and process
    image_size : int
        Size to which resize images (on which model was trained).

    Returns
    -------
    images: numpy.ndarray
        Processed images in array format with shape: (n_images, image_size, image_size, 1), values in range: [0, 1].
    images_path: list
        List of images paths
    """
    images_path = []

    for file in os.listdir(path):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            images_path.append(os.path.join(path, file))

    images = []
    for image_path in images_path:
        image = load_image(image_path, image_size)
        images.append(image)

    images = np.array(images)
    images = images.reshape(-1, image_size, image_size, 1)

    return images, images_path


def prepare_train_data(df, image_size=64):
    """
    Desciption
    ----------
    Function that loads train data from competition's parquet and processes it.


    Parameters
    ----------
    df: pd.DataFrame()
        Parquet from competition data to train model on.
    image_size: int
        Size to which resize images (on which model will be trained).

    Returns
    -------
    processed_images: numpy.ndarray
        Processed images in array format with shape: (n_images, image_size, image_size, 1), values in range: [0, 1].

    """
    processed_images = []

    for i in tqdm(range(df.shape[0])):
        image = df.iloc[i].values.reshape(137, 236)
        image = preprocess_image(image, image_size)
        processed_images.append(image)


    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
    processed_images = np.array(processed_images).reshape(-1, image_size, image_size, 1)
    return processed_images







