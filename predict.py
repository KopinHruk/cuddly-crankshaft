# Module for getting predictions based on pretrained model.
import argparse
import os
import numpy as np
import pandas as pd

from utils.preprocessing import load_image, load_images
from utils.model import create_model


def get_parser():
    """
    Description
    -----------
    Function to get arguments.


    Returns
    -------
    args: argparse.Namespace
        Module arguments.
    """
    parser = argparse.ArgumentParser(description='Module for bengali classification.')

    parser.add_argument('images_path', type=str, help='Path to image or directory with images to predict on.')
    parser.add_argument('weights_path', type=str, help='Path for loading model weights.',
                        default='weights/bengalimodal.h5', nargs='?')
    parser.add_argument('image_size', type=int, help='Size image at which the model was trained.',
                        default=64, nargs='?')

    args = parser.parse_args()

    return args


def main():

    args = get_parser()

    # Getting data to predict on
    if os.path.isfile(args.images_path):
        data = load_image(args.images_path, args.image_size)
        image_names = [args.images_path]
        data = data.reshape(-1, 64, 64, 1)
    else:
        data, image_names = load_images(args.images_path, args.image_size)

    # Creating model
    model = create_model(image_size=args.image_size, weights_path=args.weights_path)

    # Predicting
    head_root, head_vowel, head_consonant = model.predict(data, verbose=1)

    head_root = np.argmax(head_root, axis=1)
    head_vowel = np.argmax(head_vowel, axis=1)
    head_consonant = np.argmax(head_consonant, axis=1)

    # Creating and saving resulting DataFrame
    result_df = pd.DataFrame()
    result_df['image_name'] = image_names
    result_df['head_root'] = head_root
    result_df['head_vowel'] = head_vowel
    result_df['head_consonant'] = head_consonant
    result_df.to_csv('result.csv', index=False)


if __name__ == '__main__':
    main()
