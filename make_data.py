import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import os
# rdkit
def make_rdkit_imgs(df, img_size, phase='train'):
    for idx, row in tqdm(df.iterrows()):
        file = row['uid']
        smiles = row['SMILES']
        # if idx == 1:
        #     break
        m = Chem.MolFromSmiles(smiles)
        if m != None:
            img = Draw.MolToImage(m, size=(img_size, img_size))
            img_save_path = f'../data/{phase}_rdkit_imgs'
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path, exist_ok=True)
            img.save(f'{img_save_path}/{file}.png')
            
train = pd.read_csv('./data/train.csv')
dev = pd.read_csv('./data/dev.csv')
test = pd.read_csv('./data/test.csv')

train = pd.concat([train, dev], ignore_index=True)  # Train + Dev set

make_rdkit_imgs(train, img_size=300, phase='train+dev')
make_rdkit_imgs(test, img_size=300, phase='test')