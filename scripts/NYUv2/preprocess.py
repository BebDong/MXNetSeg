import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat


def write_txt(f, list_ids):
    f.write('\n'.join(list_ids))
    f.close()


def extract_data(root):
    """
    extract images and labels.
    :param root:
    :return:
    """
    print('Extracting images and labels from nyu_depth_v2_labeled.mat...')
    data = h5py.File(os.path.join(root, 'nyu_depth_v2_labeled.mat'))
    images = np.array(data['images'])
    labels = np.array(data['labels'])
    print(f'images shape: {images.shape}')
    print(f'labels shape: {labels.shape}')
    num_img = images.shape[0]
    print(f'image number: {num_img}')

    images_dir = os.path.join(root, 'images')
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    labels_dir = os.path.join(root, 'labels')
    if not os.path.isdir(labels_dir):
        os.makedirs(labels_dir)

    bar = tqdm(range(num_img))
    for i in bar:
        img = images[i]
        r = Image.fromarray(img[0]).convert('L')
        g = Image.fromarray(img[1]).convert('L')
        b = Image.fromarray(img[2]).convert('L')
        img = Image.merge('RGB', (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        img.save(os.path.join(images_dir, str(i) + '.jpg'), optimize=True)

        lbl = labels[i]
        lbl = Image.fromarray(np.uint8(lbl))
        lbl = lbl.transpose(Image.ROTATE_270)
        lbl.save(os.path.join(labels_dir, str(i) + '.png'), optimize=True)


def split(root):
    print('Generating training and validation split from split.mat...')
    split_file = loadmat(os.path.join(root, 'splits.mat'))
    train_images = tuple([int(x) for x in split_file["trainNdxs"]])
    test_images = tuple([int(x) for x in split_file["testNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    train_ids = [str(i - 1) for i in train_images]
    test_ids = [str(i - 1) for i in test_images]

    train_list_file = open(os.path.join(root, 'train.txt'), 'a')
    write_txt(train_list_file, train_ids)

    test_list_file = open(os.path.join(root, 'val.txt'), 'a')
    write_txt(test_list_file, test_ids)


def labels_40(root):
    print('Extracting labels with 40 classes from labels40.mat...')
    data = loadmat(os.path.join(root, 'labels40.mat'))
    labels = np.array(data['labels40'])
    print(f'labels shape: {labels.shape}')

    path_converted = os.path.join(root, 'labels40')
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    bar = tqdm(range(labels.shape[2]))
    for i in bar:
        label = np.array(labels[:, :, i].transpose((1, 0)))
        label_img = Image.fromarray(np.uint8(label))
        label_img = label_img.transpose(Image.ROTATE_270)
        label_img.save(os.path.join(path_converted, str(i) + '.png'), optimize=True)


def main():
    root = os.path.dirname(__file__)
    extract_data(root)
    split(root)
    labels_40(root)


if __name__ == '__main__':
    main()