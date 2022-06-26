import os
import time
import pandas as pd
import torch.nn as nn
from os.path import join as opj
from tqdm import tqdm
from easydict import EasyDict

from dataloader import *
from losses import *
from network import *


def predict(df, encoder_name, test_loader, device, model_path):
    df_submission = df.copy()
    model = SAMSUNG_RegNet_test(encoder_name).to(device)
    model.load_state_dict(torch.load(opj(model_path, 'best_model.pth'))['state_dict'])
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = torch.as_tensor(images, device=device, dtype=torch.float32)
            preds = model(images)
            preds = preds[:, 2].detach().cpu().numpy()
            preds = np.where(preds<0, 0, preds)
            preds_list.extend(list(preds))

    df_submission.iloc[:, 1] = preds_list

    return df_submission

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_test = pd.read_csv(opj('./data/', 'sample_submission.csv'))
test_img_path = opj('./data/', 'test_rdkit_imgs')
test_transform = get_train_augmentation(img_size=256, ver=1)
test_dataset = Test_dataset(df_test, test_img_path, test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# Predicting with each model.
encoder_name_reg040 = 'regnety_040'
encoder_name_reg064 = 'regnety_064'

reg040_df0 = predict(df_test, encoder_name_reg040, test_loader, device, model_path = './results/000/')
reg040_df1 = predict(df_test, encoder_name_reg040, test_loader, device, model_path = './results/001/')
reg040_df2 = predict(df_test, encoder_name_reg040, test_loader, device, model_path = './results/002/')
reg040_df3 = predict(df_test, encoder_name_reg040, test_loader, device, model_path = './results/003/')
reg040_df4 = predict(df_test, encoder_name_reg040, test_loader, device, model_path = './results/004/')
reg064_df0 = predict(df_test, encoder_name_reg064, test_loader, device, model_path = './results/005/')
reg064_df1 = predict(df_test, encoder_name_reg064, test_loader, device, model_path = './results/006/')

# ensemble.
df_test['ST1_GAP(eV)'] = 0.06*(reg040_df0.iloc[:, 1]) + 0.06*(reg040_df1.iloc[:, 1]) \
                + 0.06*(reg040_df2.iloc[:, 1]) + 0.06*(reg040_df3.iloc[:, 1]) + 0.06*(reg040_df4.iloc[:, 1]) \
                + 0.3*(reg064_df0.iloc[:, 1]) + 0.4*(reg064_df1.iloc[:, 1])

df_test.to_csv('./final_submission.csv', index=False)