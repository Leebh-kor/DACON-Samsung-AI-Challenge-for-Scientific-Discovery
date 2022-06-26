# RegNetY_040 + Cosine Annealing Scheduler
## 0Fold
python main.py --tag=regy040_0Fold --drop_path_rate=0.2 --Kfold=5 --fold=0\
                --initial_lr=1e-3 --min_lr=5e-6 --batch_size=128 --patience=100\
                --encoder_name=regnety_040 --img_size=256 --aug_ver=2\
                --scheduler=cos --tmax=295 --epochs=300 --warm_epoch=5 --exp_num=0 # --multi_gpu=True 
## 1Fold
python main.py --tag=regy040_1Fold --drop_path_rate=0.2 --Kfold=5 --fold=1\
                --initial_lr=1e-3 --min_lr=5e-6 --batch_size=128 --patience=100\
                --encoder_name=regnety_040 --img_size=256 --aug_ver=2\
                --scheduler=cos --tmax=295 --epochs=300 --warm_epoch=5 --exp_num=1
## 2Fold
python main.py --tag=regy040_2Fold --drop_path_rate=0.2 --Kfold=5 --fold=2\
                --initial_lr=1e-3 --min_lr=5e-6 --batch_size=128 --patience=100\
                --encoder_name=regnety_040 --img_size=256 --aug_ver=2\
                --scheduler=cos --tmax=295 --epochs=300 --warm_epoch=5 --exp_num=2
## 3Fold
python main.py --tag=regy040_3Fold --drop_path_rate=0.2 --Kfold=5 --fold=3\
                --initial_lr=1e-3 --min_lr=5e-6 --batch_size=128 --patience=100\
                --encoder_name=regnety_040 --img_size=256 --aug_ver=2\
                --scheduler=cos --tmax=295 --epochs=300 --warm_epoch=5 --exp_num=3
## 4Fold
python main.py --tag=regy040_4Fold --drop_path_rate=0.2 --Kfold=5 --fold=4\
                --initial_lr=1e-3 --min_lr=5e-6 --batch_size=128 --patience=100\
                --encoder_name=regnety_040 --img_size=256 --aug_ver=2\
                --scheduler=cos --tmax=295 --epochs=300 --warm_epoch=5 --exp_num=4
# RegNetY_064
## Onecycle LR Scheduler
python main.py --tag=regy064_0Fold --drop_path_rate=0.2 --Kfold=5 --fold=0\
                --initial_lr=5e-6 --max_lr=1e-3 --batch_size=128 --patience=100\
                --encoder_name=regnety_064 --img_size=256 --aug_ver=2\
                --scheduler=cycle --tmax=295 --epochs=300 --warm_epoch=5 --exp_num=5

## Charbonnier Loss + Onecycle LR Scheduler
python main.py --tag=regy064_0Fold --drop_path_rate=0.2 --Kfold=5 --fold=0\
                --initial_lr=5e-6 --max_lr=1e-3 --batch_size=128 --patience=100 --charbonnier=True\
                --encoder_name=regnety_064 --img_size=256 --aug_ver=2\
                --scheduler=cycle --tmax=295 --epochs=300 --warm_epoch=5 --exp_num=6