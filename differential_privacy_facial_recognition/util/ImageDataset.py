import os
from torch.utils.data import Dataset
from skimage import transform as trans
from sklearn import preprocessing
import cv2
import torch
from ignite.utils import to_onehot
from sklearn.preprocessing import LabelEncoder
# This class is used to store imaged and labels in a Dataset type object

class ImageDataset(Dataset):
    def __init__(self, index_file, transform, **kwargs) -> None:
        """ Create a ``ImageDataset`` object

            Args:
            index_file: image data
            transform: transform for data augmentation
        """

        super().__init__()
        self.index_file= index_file
        self.transform = transform # used to specify the types of obfuscations that will be performed on the image
        self.inputs = [] # contains the images
        self.sample_num = 0
        self.sample_nums = 0

    def parse_index_file_input(self, line):
        # take in the image path and seperate out the label from the path and return label with the path
        line_s = line.rstrip().split(' ')
        label = line_s[0]
        img_path = line_s[1]
        self.sample_num += 1
        self.class_num = self.sample_num
        return (img_path, label)

    def build_inputs(self):
        """ Read index file and saved in ``self.inputs``
        """
        num = 0
        # read in the index file that contains a list of all images and the dirs
        with open(self.index_file, 'r') as f:
            for line in f:
                sample = self.parse_index_file_input(line)
                self.inputs.append(sample)
                num += 1

        self.class_num = self.class_num + 1
        self.sample_nums = num
        self.sample_num = len(self.inputs)

    def __len__(self):
        return len(self.inputs)

    # rescale 112 x 112
    def rescale_image(self, img):
        image_size = (112, 112)
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)
        return img

    # grab the image given the file path
    def grab_image(self, path, label):
        path_str = path.rstrip()
        image = cv2.imread(path_str)
        image = self.rescale_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        return image, label

    def __getitem__(self, index):
      """ Parse image and label data from index
      """
      sample = list(self.inputs[index])
      image, label = self.grab_image(*sample)
      return image, label
