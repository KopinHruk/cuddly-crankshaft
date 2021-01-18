# Module for model training.
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.augmentator import create_datagen
from utils.preprocessing import prepare_train_data
from utils.model import create_model, lr_reduction_root, lr_reduction_vowel, \
    lr_reduction_consonant


def get_parser():
    parser = argparse.ArgumentParser(description='Module for model training.')

    parser.add_argument('n_epochs', type=int, help='Number of epochs to train model.',
                        default=16, nargs='?')
    parser.add_argument('n_parquets', type=int, help='Number of parquets to train on.',
                        default=4, nargs='?')
    parser.add_argument('batch_size', type=int, help='Model batch size.',
                        default=256, nargs='?')
    parser.add_argument('image_size', type=int, help='Size image at which the model will be trained.',
                        default=64, nargs='?')
    parser.add_argument('weights_path', type=str, help='Path for saving model weights.',
                        default='weights/bengalimodal.h5', nargs='?')

    return parser.parse_args()


def main():
    args = get_parser()

    train_df = pd.read_csv('train_data/train.csv', dtype={'grapheme_root':'uint8', 'vowel_diacritic':'uint8', 'consonant_diacritic':'uint8'})
    train_df.drop(['grapheme'], axis=1, inplace=True)

    model = create_model(image_size=args.image_size, pretrained=False)

    histories = []
    for i in range(args.n_parquets):
        print('Loading parquet.')
        parquet = pd.read_parquet(f'train_data/train_image_data_{i}.parquet')
        parquet_images_id = parquet['image_id'].values
        parquet.drop(columns='image_id', inplace=True)
        temp_df = train_df[train_df['image_id'].isin(parquet_images_id)]

        print('Transforming data.')
        all_images = prepare_train_data(parquet)
        Y_train_root = pd.get_dummies(temp_df['grapheme_root']).values
        Y_train_vowel = pd.get_dummies(temp_df['vowel_diacritic']).values
        Y_train_consonant = pd.get_dummies(temp_df['consonant_diacritic']).values

        x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(
            all_images, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)

        datagen = create_datagen(x_train)

        print('Training model.')
        history = model.fit_generator(
            datagen.flow(x_train, {'dense_2': y_train_root, 'dense_3': y_train_vowel, 'dense_4': y_train_consonant}, batch_size=args.batch_size),
            epochs=args.n_epochs,
            validation_data=(x_test, [y_test_root, y_test_vowel, y_test_consonant]),
            steps_per_epoch=x_train.shape[0] // args.batch_size,
            callbacks=[lr_reduction_root, lr_reduction_vowel, lr_reduction_consonant]
        )

        histories.append(history)

    model.save(args.weights_path)


if __name__ == '__main__':
    main()
