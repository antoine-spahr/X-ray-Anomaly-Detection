import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as TF
from torch.utils import data
from sklearn.model_selection import train_test_split

from transforms import HistEqualization, AutoContrast, ResizeMax, PadToSquare, MinMaxNormalization

class MURA_Dataset(data.Dataset):
    """

    """
    def __init__(self, sample_df, data_path, load_semilabels=True):
        """

        """
        data.Dataset.__init__(self)
        self.sample_df = sample_df
        self.data_path = data_path
        self.load_semilabels = load_semilabels
        self.transform = TF.Compose([TF.Grayscale(), \
                                     AutoContrast(cutoff=1), \
                                     TF.RandomHorizontalFlip(p=0.5), \
                                     TF.RandomVerticalFlip(p=0.5), \
                                     TF.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0), \
                                     TF.RandomAffine(0, scale=(0.8,1.2)), \
                                     TF.RandomRotation((-20,20)), \
                                     ResizeMax(512), \
                                     PadToSquare(), \
                                     MinMaxNormalization(), \
                                     TF.ToTensor()])

    def __len__(self):
        """

        """
        return self.sample_df.shape[0]

    def __getitem__(self, idx):
        """
        
        """
        # load image, label
        im = Image.open(self.data_path + df.loc[idx,'filename'])#skimage.io.imread(self.data_path + self.sample_df.loc[idx,'filename'])
        label = torch.tensor(self.sample_df.loc[idx,'abnormal_XR'])
        # apply given transform
        im = self.transform(im)

        if self.load_semilabels:
            semi_label = torch.tensor(self.sample_df.loc[idx, 'semi_label'])
            return im, label, semi_label
        else:
            return im, label


class MURA_TrainValidTestSplitter:
    """
    --> init : creat object
    --> split_data : generate the subseet with the semi-supervised labels
    """
    def __init__(self, data_info, train_frac=0.5,
                 ratio_known_normal=0.0, ratio_known_abnormal=0.0, random_state=42):
        """
        Constructor of the splitter.
        ----------
        INPUT
            |---- data_info (pd.DataFrame) the whole data with columns generated
            |           with generate_data_info function
            |---- train_frac (float) define the fraction of the data to use as
            |           train set. The train set is mostly composed of normal
            |           sample. There must thus be enough of them.
            |---- ratio_known_normal (float) the fraction of knwon normal samples
            |---- ratio_known_abnormal (float) the fraction of knwon abnormal samples
            |---- random_state (int) the seed for reproducibility
        OUTPUT
            |---- None
        """
        # input
        self.subsets = {}
        self.data_info = data_info
        assert train_frac <= 1, f'Input Error. The train fraction must larger than one. Here it is {train_frac}'
        self.train_frac = train_frac
        assert ratio_known_normal <= 1, f'Input Error. The ratio_known_normal must be smaller than one. Here it is {ratio_known_normal}'
        self.ratio_known_normal = ratio_known_normal
        assert ratio_known_abnormal <= 1, f'Input Error. The ratio_known_abnormal must be smaller than one. Here it is {ratio_known_abnormal}'
        self.ratio_known_abnormal = ratio_known_abnormal
        self.random_state = random_state

    def split_data(self):
        """
        Split the MURA dataset into a train, validation and test sets. To avoid
        test leakage, the split is made at the level of patients bodypart (all
        XR from a patient's hand will be on the same set).
        1) The train contains train_frac samples, of which ratio_known_abnormal
        are abnormal and the rest are normal XR. All the abnormal train XR are
        considered as known.
        2) the rest of the normal/mixt and abnormal are equally shared between
        the validation and test set. Mixt patient bodypart (patient hand with both
        normal and abnormal XR) are considered abnormal for the spliting.
        3) For each set, a fraction of normal and abnormal (in the case of the
        train, all abnormal are considered as known) are labeled. The resulting
        semi-supervised labelling is : 0 = unknown ; 1 = known normal ;
        -1 = known abnormal
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """
        # group sample by patient and body part
        tmp = self.data_info.groupby(['patientID', 'body_part']).max()
        # get the index (i.e. patient and bodypart) where none of the body part XR of a given patient are abnormal
        idx_list_normal = tmp[tmp.body_part_abnormal == 0].index.to_list()
        # get the index (i.e. patient and bodypart) where at least one but not all of the body part XR of a given patient are abnormal
        idx_list_mixt = tmp[tmp.body_part_abnormal == 0.5].index.to_list()
        # get the index (i.e. patient and bodypart) where all one of the body part XR of a given patient are abnormal
        idx_list_abnormal = tmp[tmp.body_part_abnormal == 1].index.to_list()
        total = len(idx_list_normal)+len(idx_list_mixt)+len(idx_list_abnormal)
        train_size = self.train_frac*total
        assert train_size < len(idx_list_normal), f'There are not enough normal sample for the given train_frac : {self.train_frac}. \
                                                    There are {len(idx_list_normal)} normal sample over {total} total samples.'
        valid_size = (1-self.train_frac)*0.5*total
        test_size = (1-self.train_frac)*0.5*total
        # randomly pick (1-ratio_known_abnormal)*train_frac*total from the normal index for the train set
        train_idx_normal, remain = train_test_split(idx_list_normal, \
                                                    train_size=int((1-self.ratio_known_abnormal)*train_size),\
                                                    random_state=self.random_state)
        # split the rest equally in the validation and test set
        valid_idx_normal, test_idx_normal = train_test_split(remain, test_size=0.5, random_state=self.random_state)
        # add ratio_known_abnormal*train_frac*total from the abnormal index
        if self.ratio_known_abnormal == 0.0:
            train_idx_abnormal, remain = [], idx_list_abnormal
        else:
            train_idx_abnormal, remain = train_test_split(idx_list_abnormal, \
                                                          train_size=int(self.ratio_known_abnormal*train_size),\
                                                          random_state=self.random_state)
        # split the rest equally in the validation and test set
        valid_idx_abnormal, test_idx_abnormal = train_test_split(remain, test_size=0.5, random_state=self.random_state)
        # split the mixt between test and validation and consider them as abnormal patients bodypart
        valid_idx_mixt, test_idx_mixt = train_test_split(idx_list_mixt, test_size=0.5, random_state=42)
        valid_idx_abnormal += valid_idx_mixt
        test_idx_abnormal += test_idx_mixt
        # get the known and unknown index for each sets
        # get a fraction of normal known
        if self.ratio_known_normal == 0.0:
            train_idx_known, train_idx_unknown = [], train_idx_normal
            valid_idx_known, valid_idx_unknown = [], valid_idx_normal
            test_idx_known, test_idx_unknown = [], test_idx_normal
        else:
            train_idx_known, train_idx_unknown = train_test_split(train_idx_normal, \
                                                            train_size=int(self.ratio_known_normal*train_size),\
                                                            random_state=self.random_state)
            valid_idx_known, valid_idx_unknown = train_test_split(valid_idx_normal, \
                                                            train_size=int(self.ratio_known_normal*valid_size),\
                                                            random_state=self.random_state)
            test_idx_known, test_idx_unknown = train_test_split(test_idx_normal, \
                                                            train_size=int(self.ratio_known_normal*test_size), \
                                                            random_state=self.random_state)
        # get the abnormal known
        # all abnormal in train are known
        train_idx_known += train_idx_abnormal
        if self.ratio_known_abnormal == 0.0:
            valid_idx_unknown += valid_idx_abnormal
            test_idx_unknown += test_idx_abnormal
        else:
            valid_idx_known_abnormal, valid_idx_unknown_abnormal = train_test_split(valid_idx_abnormal, \
                                                                        train_size=int(self.ratio_known_abnormal*valid_size), \
                                                                        random_state=self.random_state)
            valid_idx_known += valid_idx_known_abnormal
            valid_idx_unknown += valid_idx_unknown_abnormal
            test_idx_known_abnormal, test_idx_unknown_abnormal = train_test_split(test_idx_abnormal, \
                                                                        train_size=int(self.ratio_known_abnormal*test_size),\
                                                                        random_state=self.random_state)
            test_idx_known += test_idx_known_abnormal
            test_idx_unknown += test_idx_unknown_abnormal

        # get the subsample dataframe with semi-label
        train_df = self.generate_semisupervized_label(train_idx_known, train_idx_unknown)
        valid_df = self.generate_semisupervized_label(valid_idx_known, valid_idx_unknown)
        test_df = self.generate_semisupervized_label(test_idx_known, test_idx_unknown)
        # shuffle the dataframes
        self.subsets['train'] = train_df.sample(frac=1).reset_index(drop=True)
        self.subsets['valid'] = valid_df.sample(frac=1).reset_index(drop=True)
        self.subsets['test'] = test_df.sample(frac=1).reset_index(drop=True)

    def generate_semisupervized_label(self, idx_known, idx_unknown):
        """
        Assigne the semi-supervized labels at the given indices
        0 = unknown ; 1 = known normal ; -1 = known abnormal
        ----------
        INPUT
            |---- idx_known (list of tuple) the multiindex (patient, bodypart)
            |           of the known elements
            |---- idx_unknown (list of tuple) the multiindex (patient, bodypart)
            |           of the unknown elements
        OUTPUT
            |---- df (pd.DataFrame) the dataframe at the passed index with semi-
            |           supervised labels
        """
        tmp_df = self.data_info.set_index(['patientID','body_part'])
        # associate semi-supervized settings
        if len(idx_known) > 0:
            df_known = tmp_df.loc[idx_known,:]
            df_known['semi_label'] = df_known.abnormal_XR.apply(lambda x: -1 if x==1 else 1)
            df_unknown = tmp_df.loc[idx_unknown,:]
            df_unknown['semi_label'] = 0
            return pd.concat([df_known, df_unknown], axis=0).reset_index()
        else:
            df_unknown = tmp_df.loc[idx_unknown,:]
            df_unknown['semi_label'] = 0
            return df_unknown.reset_index()

    def get_subset(self, name):
        """
        Return the data subset requested.
        ----------
        INPUT
            |---- name (str) the subset name : 'train', 'valid' or 'test'
        OUTPUT
            |---- subset (pd.DataFrame) the subset dataframe with semi-lables
        """
        assert name in ['train', 'valid', 'test'], f'Invalid dataset name! {name} has been provided but must be one of [train, valid, test]'
        return self.subsets[name]

    def print_stat(self):
        """

        """
        # TO DO print a summary of the split


# %% ###########################################################################
################################ EXAMPLE OF USAGE ##############################
################################################################################
import matplotlib.pyplot as plt

DATA_PATH = '../../../data/'
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)

spliter = MURA_TrainValidTestSplitter(df, train_frac=0.5, ratio_known_normal=0.05, ratio_known_abnormal=0.05)
spliter.split_data()

train_df = spliter.get_subset('train')
valid_df = spliter.get_subset('valid')
test_df = spliter.get_subset('test')

datasetMURA = MURA_Dataset(train_df, data_path=DATA_PATH+'PROCESSED/')
image_test, label, semi_label = datasetMURA.__getitem__(12)

fig, ax = plt.subplots(1,1,figsize=(8,8))
ax.set_title('Transformed sample from the MURA dataset')
ax.imshow(image_test[0,:,:], cmap='Greys_r')
plt.show()
