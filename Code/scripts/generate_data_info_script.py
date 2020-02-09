import pandas as pd

sys.path.append('../')
from src.preprocessing.get_data_info import generate_data_info

DATA_PATH = '../data/'

def main():
    """

    """
    df1 = generate_data_info(DATA_PATH+'RAW/train_image_paths.csv')
    df2 = generate_data_info(DATA_PATH+'RAW/valid_image_paths.csv')
    df = pd.concat([df1,df2], axis=0).reset_index()
    df.to_csv(DATA_PATH+'data_info.csv')

if __name__ == '__main__':
    print(__doc__)
    main()
