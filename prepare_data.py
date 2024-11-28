import pandas as pd
import warnings
import tqdm
import os
import shutil
from utils import CLASSES

warnings.filterwarnings('ignore', category=Warning)


data_path = 'Data/source/images'
data_save_path = 'Data/data'
meta_df = pd.read_csv('Data/source/metadata.csv')
meta_df_covid = meta_df[meta_df['finding'] == 'COVID-19']
meta_df_covid['survival'] = meta_df_covid['survival'].fillna('N')

neg_path = os.path.join(data_save_path, CLASSES[0])
pos_path = os.path.join(data_save_path, CLASSES[1])

os.makedirs(neg_path, exist_ok=True)
os.makedirs(pos_path, exist_ok=True)

required_cols = ['survival', 'filename', 'modality', 'view']

for row in tqdm.tqdm(meta_df_covid[required_cols].values, desc='[INFO] Preparing Images'):
    if row[2] == 'X-ray':
        if row[3] in ['PA', 'AP']:
            image_path = os.path.join(data_path, row[1])
            image_save_path = os.path.join(neg_path if row[0] == 'N' else pos_path, row[1])
            if os.path.isfile(image_path):
                shutil.copy(image_path, image_save_path)
