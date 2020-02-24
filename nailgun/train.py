import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import albumentations

from nailgun import data, model


class AugmentedGenerator(Sequence):
    """Helper class that generates augmented images for model training."""

    def __init__(self, x, y, batch_size, augmentations):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return model.preprocess_image(
            np.stack([self.augment(image=x)['image'] for x in batch_x], axis=0)
        ), np.array(batch_y)


def train(args):
    images, y = data.load(args.data_dir)

    X = np.array([model.crop_nail(image, height=320, width=320) for image in images])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=0)
    net = model.create()

    X_test = np.array([model.resize(image, net.input_shape[1:3]) for image in X_test])
    X_test = model.preprocess_image(X_test)

    augmentator = albumentations.Compose([
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=.8),
        albumentations.OneOf([
            albumentations.RandomRotate90(),
            albumentations.Flip(),
        ], p=.6),
        albumentations.Resize(width=net.input_shape[2], height=net.input_shape[1]),
        albumentations.GaussNoise(var_limit=(1, 3)),
    ])

    net.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy'])

    net.fit_generator(
        AugmentedGenerator(X_train, y_train, 40, augmentator),
        epochs=args.epochs,
        validation_data=(X_test, y_test),
    )

    net.save_weights(args.target, overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='model')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--test-size', default=28, type=int)
    parser.add_argument('--data-dir', default='data')

    args = parser.parse_args()
    train(args)
