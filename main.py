import os
import pprint
import random
import warnings
# import neptune
import torch
import numpy as np
from config import getConfig
from trainer import Trainer
warnings.filterwarnings('ignore')
args = getConfig()

def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    save_path = os.path.join(args.model_path, (args.exp_num).zfill(3))

    if args.logging:
        api = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmNhYzY4OC1mOGQ5LTRmOGEtYTEzNy0xNTY1NWNkMjNjNDEifQ=='
        neptune.init('qlscm777/material', api_token=api)
        temp = neptune.create_experiment(name=args.experiment, params=vars(args))
        experiment_num = str(temp).split('-')[-1][:-1]  # 모델저장경로 설정위한 Experiment Number받기
        neptune.append_tag(args.tag)
        # experiment_num = '1'
        save_path = os.path.join(args.model_path, experiment_num.zfill(3))

    # Create model directory
    os.makedirs(save_path, exist_ok=True)
    Trainer(args, save_path)

if __name__ == '__main__':
    main(args)
