import os
import sys
from glob import glob
import skimage
sys.path.append('../')

from src.utils.utils import print_progessbar, print_param_summary
from src.preprocessing.cropping_rect import find_squares, crop_squares

IN_DATA_PATH = '../data/RAW/'
OUT_DATA_PATH = '../data/PROCESSED/'

def main():
    """

    """
    print('-'*100 + '\n' + 'PREPROCESSING'.center(100) + '\n' + '-'*100)
    summary = {}
    # iterate body part
    for bpdir in glob(IN_DATA_PATH + '/*/'):
        bpname = bpdir.split("/")[-2][3:].title()
        print(f'|--- Processing {bpname} X-rays')
        XR_count = 0
        patient_count = 0
        missing_patients = 0
        uncropped_img = 0
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
                    # get squares
                    squares = find_squares(img, min_area=50000)

                    if squares:
                        warped = crop_squares(squares, img)
                        skimage.io.imsave(write_dir + "/" + os.path.basename(fn), warped)
                    else:
                        skimage.io.imsave(write_dir + "/" + os.path.basename(fn), img)
                        uncropped_img += 1

            print_progessbar(i, n_dir, Name='|------ Patients', Size=50)
            if not has_img: missing_patients += 1
        summary[bpname] = {'X-ray':XR_count, 'patient':patient_count, \
                           'missing patient':missing_patients, 'uncropped images':uncropped_img}
        print('')
    print_param_summary(**summary)

if __name__ == '__main__':
    print(__doc__)
    main()
