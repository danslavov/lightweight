"""Elements Dataloader"""
import os
import random
from collections import OrderedDict

from cv2 import cv2 as cv2
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['ElementSegmentation']

from matplotlib import pyplot as plt


class ElementSegmentation(data.Dataset):
    """Elements Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to Elements folder. Default is './datasets/elements'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'elements'
    NUM_CLASS = 4
    PATH = 'C:/Users/Admin/PycharmProjects/ENet-PyTorch-davidtvs/data/Elements'
    # PATH = 'C:/Users/Admin/PycharmProjects/lightweight/datasets/Elements-one'  # run over 1 image for debugging

    def __init__(self, root=PATH, split='train', mode=None, transform=None,
                 base_size=800, crop_size=800, **kwargs):
        super(ElementSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
        #                       23, 24, 25, 26, 27, 28, 31, 32, 33]  # INFO: orig
        self.valid_classes = [0, 1, 2, 3]  # INFO: mine
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):

        # cv2.imwrite('C:/Users/Admin/Desktop/tmp/mask.png', mask)

        # plt.imshow(mask, interpolation='nearest')
        # plt.show()

        # INFO: mine -- convert the 3-channel (RGB) mask to 0-channel "class map"
        colors = OrderedDict([
            ('background', (170, 170, 170)),  # 0   black
            ('capacitor', (0, 0, 255)),  # 1        blue
            ('capacitor-flat', (255, 0, 0)),  # 2   red
            ('resistor', (255, 255, 0)),  # 3       yellow
        ])
        shape = (mask.shape[0], mask.shape[1])
        class_map = np.zeros(shape, np.int32)  # all pixels are background (value 0)
        for class_number in range(len(colors)):
            if class_number == 0: continue  # no need to change anything, since the background is already a 0
            color = list(colors.items())[class_number][1]
            red_value = color[0]
            green_value = color[1]
            blue_value = color[2]

            cond_red = mask[:, :, 0] == red_value
            cond_green = mask[:, :, 1] == green_value
            cond_blue = mask[:, :, 2] == blue_value

            cond_matrix = np.logical_and(np.logical_and(cond_red, cond_blue), cond_green)
            # cond_matrix = torch.unsqueeze(cond_matrix, 0)  # add the channel dimension
            class_map[cond_matrix] = class_number

        # cv2.imwrite('C:/Users/Admin/Desktop/tmp/class_map.png', class_map)

        # INFO: mine. Skip the following checks and transformations
        # values = np.unique(class_map)
        # for value in values:
        #     assert (value in self._mapping)
        # index = np.digitize(class_map.ravel(), self._mapping, right=True)
        # return self._key[index].reshape(class_map.shape)
        return class_map

    # TODO: all transforms are skipped (except ToTensor and RGB-to-class), so reduce the code
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = img.copy()  # INFO: to fix 'negative strides' error
            img = self.transform(img)
        return img, mask

    # INFO: mine.
    # All augmentation transforms are skipped. TODO: Resizing is also skipped; add it, if needed.
    def _val_sync_transform(self, img, mask):
        # outsize = self.crop_size
        # short_size = outsize
        # w, h = img.size
        # if w > h:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # # center crop
        # w, h = img.size
        # x1 = int(round((w - outsize) / 2.))
        # y1 = int(round((h - outsize) / 2.))
        # img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        # mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    # INFO: mine.
    # All augmentation transforms are skipped. TODO: Resizing is also skipped; add it, if needed.
    def _sync_transform(self, img, mask):
        # random mirror
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # crop_size = self.crop_size
        # random scale (short edge)
        # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # w, h = img.size
        # if h > w:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # else:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        # if short_size < crop_size:
        #     padh = crop_size - oh if oh < crop_size else 0
        #     padw = crop_size - ow if ow < crop_size else 0
        #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        # w, h = img.size
        # x1 = random.randint(0, w - crop_size)
        # y1 = random.randint(0, h - crop_size)
        # img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))
        # final transform

        # img.save('C:/Users/Admin/Desktop/tmp/img.png', format='png')
        # mask.save('C:/Users/Admin/Desktop/tmp/mask.png', format='png')

        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        arr = np.array(img)  # Numpy changes color order from RGB to BGR, so
        arr = arr[:, :, ::-1]  # change from BGR to RGB
        return arr

    def _mask_transform(self, mask):
        # mask = cv2.imread('C:/Users/Admin/Desktop/red_green_blue.png')
        # mask.save('C:/Users/Admin/Desktop/tmp/begin.png', format='png')
        arr = np.array(mask)#.astype('int32')  # Numpy changes color order from RGB to BGR, so
        # cv2.imwrite('C:/Users/Admin/Desktop/tmp/BGR.png', arr)
        # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)  # change from BGR to RGB
        # cv2.imwrite('C:/Users/Admin/Desktop/tmp/RGB.png', arr)
        target = self._class_to_index(arr)

        # result = torch.LongTensor(np.array(target).astype('int32'))
        # cv2.imwrite('C:/Users/Admin/Desktop/tmp/result.png', result)

        # TODO: 'target' is already ndarray, so maybe skip using 'np.array(target)'
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    # maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')  # INFO: orig
                    maskname = filename
                    # maskpath = os.path.join(mask_folder, foldername, maskname)  # INFO: orig, duplicated foldername
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        img_folder = os.path.join(folder, split)
        mask_folder = os.path.join(folder, split + '_labels')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = ElementSegmentation()
    img, label = dataset[0]
