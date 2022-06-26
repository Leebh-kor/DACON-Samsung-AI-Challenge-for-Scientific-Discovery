import os
import time
# import neptune
import logging
import pandas as pd
import torch.nn as nn
import torch_optimizer as optim
from os.path import join as opj
from tqdm import tqdm
from ptflops import get_model_complexity_info
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, grad_scaler

import utils
from dataloader import *
from losses import *
from network import *

class Trainer():
    def __init__(self, args, save_path):
        '''
        args: arguments
        save_path: Model 가중치 저장 경로
        '''
        super(Trainer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging
        log_file = os.path.join(save_path, 'log.log')
        self.logger = utils.get_root_logger(logger_name='IR', log_level=logging.INFO, log_file=log_file)
        self.logger.info(args)
        self.logger.info(args.tag)

        # Train, Valid Set load
        df_train = pd.read_csv(opj(args.data_path, 'train.csv'))
        df_train = df_train.drop([1688, 14782, 28906, 29068], axis=0).reset_index(drop=True)
        df_dev = pd.read_csv(opj(args.data_path, 'dev.csv'))

        # Split Fold
        kf = KFold(n_splits=args.Kfold, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(df_train)))):
            df_train.loc[val_idx, 'fold'] = fold
        val_idx = list(df_train[df_train['fold'] == int(args.fold)].index)

        df_val = df_train[df_train['fold'] == args.fold].reset_index(drop=True)
        df_train = df_train[df_train['fold'] != args.fold].reset_index(drop=True)

        # Concat Dev
        df_train = pd.concat([df_train, df_dev]).reset_index(drop=True)

        trn_img_path = opj(args.data_path, 'train+dev_rdkit_imgs')

        # Use Unlabeled data
        if args.use_ssl_df is not None:
            df_unlabel = pd.read_csv(opj(args.data_path, 'pseudo_label_version', f'{args.use_ssl_df}.csv'))
            filter_idx = np.where((df_unlabel['S1_energy(eV)'] <= 0) | (df_unlabel['T1_energy(eV)'] <= 0))[0]
            df_unlabel = df_unlabel.drop(filter_idx, axis=0)
            print(f'Number of Additional Unlabeled data:{len(df_unlabel)}')
            df_train = pd.concat([df_train, df_unlabel], ignore_index=True)
            trn_img_path = opj(args.data_path, 'train+unlabel_imgs/')

        # Augmentation
        self.train_transform = get_train_augmentation(img_size=args.img_size, ver=args.aug_ver)
        self.test_transform = get_train_augmentation(img_size=args.img_size, ver=1)

        # TrainLoader
        self.train_loader = get_loader(df_train, trn_img_path, phase='train', batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, transform=self.train_transform)
        self.val_loader = get_loader(df_val, trn_img_path, phase='train', batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, transform=self.test_transform)

        # Network
        if args.model_type == 'Transformer':
            self.model = SAMSUNG_Transformer(args).to(self.device)
            print('Transformer Architecture')
        elif args.model_type == 'CNNV2':
            self.model = SAMSUNG_RegNet(args).to(self.device)
            # self.model.apply(init_weights)
            print('CNN V2 Architecture')

        self.logger.info('Using the only image model')
        macs, params = get_model_complexity_info(self.model, (3, args.img_size, args.img_size), as_strings=True,
                                                 print_per_layer_stat=False, verbose=False)
        self.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        self.logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # Loss
        self.criterion = CharbonnierLoss() if args.charbonnier else nn.L1Loss()
        
        # Optimizer & Scheduler
        self.optimizer = optim.Lamb(self.model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
        
        iter_per_epoch = len(self.train_loader)
        self.warmup_scheduler = utils.WarmUpLR(self.optimizer, iter_per_epoch * args.warm_epoch)

        if args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestone, gamma=args.lr_factor, verbose=True)
        elif args.scheduler == 'cos':
            tmax = args.tmax # half-cycle 
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = tmax, eta_min=args.min_lr, verbose=True) # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs-args.warm_epoch, eta_min=args.min_lr)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.max_lr, steps_per_epoch=iter_per_epoch, epochs=args.epochs)

        if args.multi_gpu:
            self.model = nn.DataParallel(self.model).to(self.device)

        load_epoch=0
        if args.re_training_exp is not None:
            pth_files = torch.load(f'./results/{args.re_training_exp}/best_model.pth')
            load_epoch = pth_files['epoch']
            self.model.load_state_dict(pth_files['state_dict'])
            self.optimizer.load_state_dict(pth_files['optimizer'])
            print(f'Start {load_epoch+1} Epoch Re-training')
            for i in range(args.warm_epoch+1, load_epoch+1):
                self.scheduler.step()

        # Train / Validate
        best_mae = np.inf
        best_epoch = 0
        early_stopping = 0
        start = time.time()
        for epoch in range(load_epoch+1, args.epochs+1):
            self.epoch = epoch

            if args.scheduler == 'cos':
                if epoch > args.warm_epoch:
                    self.scheduler.step()

            # Training
            train_loss, train_mae1, train_mae2 = self.training(args)

            # Model weight in Multi_GPU or Single GPU
            state_dict= self.model.module.state_dict() if args.multi_gpu else self.model.state_dict()

            # Validation
            val_loss, val_mae1, val_mae2 = self.validate(args, phase='val')

            if args.logging == True:
                neptune.log_metric('Train loss', train_loss)
                neptune.log_metric('Train MAE1', train_mae1)
                neptune.log_metric('Train MAE2', train_mae2)   

                neptune.log_metric('val loss', val_loss)
                neptune.log_metric('val MAE1', val_mae1)
                neptune.log_metric('val MAE2', val_mae2)   

            # Save models
            if val_mae1 < best_mae:
                early_stopping = 0
                best_epoch = epoch
                best_mae = val_mae1

                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict()
                    }, os.path.join(save_path, 'best_model.pth'))
                self.logger.info(f'-----------------SAVE:{best_epoch}epoch----------------')
            else:
                early_stopping += 1

            # Early Stopping
            if early_stopping == args.patience:
                break

            if (epoch % 10 == 0) and (epoch > 280):
                torch.save({'epoch':epoch,
                            'state_dict':state_dict,
                            'optimizer': self.optimizer.state_dict()
                    }, os.path.join(save_path, f'{epoch}epoch.pth'))

        self.logger.info(f'\nBest Val Epoch:{best_epoch} | Val MAE:{best_mae:.4f}')
        end = time.time()
        self.logger.info(f'Total Process time:{(end - start) / 60:.3f}Minute')
        # neptune.stop()

    # Training
    def training(self, args):
        self.model.train()
        train_loss = utils.AvgMeter()
        train_mae1 = utils.AvgMeter()
        train_mae2 = utils.AvgMeter()

        # 몇 에폭동안 Linear만 학습 (Linear가 아닌 레이어는 고정)
        if self.epoch <= args.freeze_epoch:
            self.model.eval()
            for name, params in self.model.named_parameters():
                if ('last' in name) or ('head') in name:
                    params.requires_grad = True
                    print(name)

        scaler = grad_scaler.GradScaler()
        for i, (images, targets) in enumerate(tqdm(self.train_loader)):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            targets = torch.tensor(targets, device=self.device, dtype=torch.float32)
            
            if self.epoch <= args.warm_epoch:
                self.warmup_scheduler.step()

            self.optimizer.zero_grad()
            if args.amp:
                with autocast():
                    preds = self.model(images)
                    loss = self.criterion(preds, targets)
                scaler.scale(loss).backward()

                # Gradient Clipping
                if args.clipping is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)

                scaler.step(self.optimizer)
                scaler.update()

            else:
                preds = self.model(images)
                loss = self.criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), args.clipping)
                self.optimizer.step()

            if args.scheduler == 'cycle':
                if self.epoch > args.warm_epoch:
                    self.scheduler.step()

            # Metric
            mae1 = torch.mean(torch.abs(targets[:, 2] - preds[:, 2]))  # S1-T1 예측
            mae2 = torch.mean(torch.abs(targets[:, 2] - (preds[:, 0] - preds[:, 1])))  # S1, T1 각각 예측한 것을 뺀 것

            # log
            train_loss.update(loss.item(), n=images.size(0))
            train_mae1.update(mae1.item(), n=images.size(0))
            train_mae2.update(mae2.item(), n=images.size(0))

        self.logger.info(f'Epoch:[{self.epoch:03d}/{args.epochs:03d}]')
        self.logger.info(f'Train Loss:{train_loss.avg:.3f} | mae1:{train_mae1.avg:.3f} | mae2:{train_mae2.avg:.3f}')
        return train_loss.avg, train_mae1.avg, train_mae2.avg
            
    # Validation or Dev
    def validate(self, args, phase='val'):
        self.model.eval()
        with torch.no_grad():
            val_loss = utils.AvgMeter()
            val_mae1 = utils.AvgMeter()
            val_mae2 = utils.AvgMeter()
            loader = self.val_loader if phase=='val' else self.dev_loader

            for i, (images, targets) in enumerate(loader):
                images = torch.tensor(images, device=self.device, dtype=torch.float32)
                targets = torch.tensor(targets, device=self.device, dtype=torch.float32)

                preds = self.model(images)
                loss = self.criterion(preds, targets)

                # Metric
                mae1 = torch.mean(torch.abs(targets[:, 2] - preds[:, 2]))  # S1-T1 예측
                mae2 = torch.mean(torch.abs(targets[:, 2] - (preds[:, 0] - preds[:, 1])))  # S1, T1 각각 예측한 것을 뺀 것

                # log
                val_loss.update(loss.item(), n=images.size(0))
                val_mae1.update(mae1.item(), n=images.size(0))
                val_mae2.update(mae2.item(), n=images.size(0))

            self.logger.info(f'{phase} Loss:{val_loss.avg:.3f} | mae1:{val_mae1.avg:.3f} | mae2:{val_mae2.avg:.3f}')
        return val_loss.avg, val_mae1.avg, val_mae2.avg

