import os
import sys
from glob import glob
import warnings
import pandas as pd
import skimage

sys.path.append('../')
from src.utils.utils import print_progessbar, print_param_summary
from src.preprocessing.cropping_rect import find_squares, crop_squares
from src.preprocessing.segmentation import find_best_mask

IN_DATA_PATH = '../../data/RAW/'
OUT_DATA_PATH = '../../data/PROCESSED/'
DATAINFO_PATH = '../../data/'

def main():
    """

    """
    print('-'*100 + '\n' + 'PREPROCESSING'.center(100) + '\n' + '-'*100)
    summary = {}
    # load the data info dataframe
    df_info = pd.read_csv(DATAINFO_PATH + 'data_info.csv')
    df_info = df_info.drop(df_info.columns[0:2], axis=1)
    # add cropped and low_contrast columns of zeros
    df_info['uncropped'] = 0
    df_info['low_contrast'] = 0
    df_info['mask_filename'] = ''
    # the min area per body part
    min_areas = {'Hand':50000, 'Elbow':50000, 'Finger':30000, 'Forearm':50000, \
                 'Humerus':50000, 'Shoulder':50000, 'Wrist':50000}
    # iterate body part
    for bpdir in glob(IN_DATA_PATH + '/*/'):
        bpname = bpdir.split("/")[-2][3:].title()
        print(f'|--- Processing {bpname} X-rays')
        XR_count = 0
        patient_count = 0
        missing_patients = 0
        uncropped_img = 0
        low_contrast = 0
        # iterate patient
        n_dir = len(glob(bpdir + '/*/'))
        for i, pdir in enumerate(glob(bpdir + '/*/')):
            has_img = False
            patient_count += 1
            # iterate over studies
            for sdir in glob(pdir + '/*/'):
                # create the write dir if it does not exist
                write_dir = OUT_DATA_PATH + '/'.join(sdir.split('/')[-4:-1])
                if not os.path.isdir(write_dir):
                    os.makedirs(write_dir)
                # iterate over images
                for fn in glob(sdir + '/*'):
                    XR_count += 1
                    has_img = True
                    img = skimage.io.imread(fn)
                    ######################## CROP ##############################
                    # get squares
                    squares = find_squares(img, min_area=min_areas[bpname])
                    # crop square
                    if squares:
                        img = crop_squares(squares, img)
                    else:
                        uncropped_img += 1
                        df_info.loc[df_info.filename == '/'.join(fn.split('/')[-4:]), 'uncropped'] = 1
                    # save image if image is not low contrast (i.e. only black)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        skimage.io.imsave(write_dir + "/" + os.path.basename(fn), img)
                    # add info to dataframe
                    if skimage.exposure.is_low_contrast(img):
                        low_contrast += 1
                        df_info.loc[df_info.filename == '/'.join(fn.split('/')[-4:]), 'low_contrast'] = 1

                    ##################### Segmentation #########################
                    mask = find_best_mask(img)
                    # save mask
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        skimage.io.imsave(write_dir + "/" + 'mask_' + os.path.basename(fn), mask)
                        df_info.loc[df_info.filename == '/'.join(fn.split('/')[-4:]), 'mask_filename'] = '/'.join(fn.split('/')[-4:-1]) + 'mask_' + os.path.basename(fn)

            print_progessbar(i, n_dir, Name='|------ Patients', Size=50)
            if not has_img: missing_patients += 1

        summary[bpname] = {'X-ray':XR_count, 'patient':patient_count, \
                           'missing patient':missing_patients, 'uncropped images':uncropped_img, \
                           'low contrast image':low_contrast}
        print('')
    print_param_summary(**summary)
    # save the updated data_info
    df_info.to_csv(DATAINFO_PATH + 'data_info.csv')

if __name__ == '__main__':
    print(__doc__)
    main()
