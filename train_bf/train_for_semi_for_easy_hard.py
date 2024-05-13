import os, tempfile
import random
import torch
import wandb
import numpy as np
import argparse as ap
import torch.nn as nn
import monai
from core.call_data import *
from core.call_model import *
from core.core import train
from core.train_search import main_search
from config.call import call_config
from transforms.call import *
from core.core_for_semi import *
from core.core_for_semi_for_easy_hard import *
# from torch.utils.data.dataset import ConcatDataset
join = os.path.join
import itertools
import warnings

# torch.multiprocessing.set_start_method('spawn')

def main(info, config, logging=False):
    if logging:
        run = wandb.init(project=info["PROJ_NAME"], entity=info["ENTITY"]) 
        wandb.config.update(config) 

    torch.manual_seed(config["SEEDS"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["SEEDS"])
    
    # Dataset
    datasets = os.path.join(info["ROOT"], 'dataset.json')
    file_list = load_decathlon_datalist(datasets, True, 'training')
    train_list, valid_list = call_fold_dataset(file_list, target_fold=config["FOLD"], total_folds=info["FOLDS"])  

    un_path = info["ROOT_unlabeled"]
    img_names = sorted(os.listdir(un_path))
    unlabel_list = [
        {"image": join(un_path, name), "label": 'nan'} for name in img_names
    ]

    print('Train', len(train_list), 'Valid', len(valid_list))
    if logging:
        artifact = wandb.Artifact(
            "dataset", type="dataset", 
            metadata={"train_list":train_list, "valid_list":valid_list, "train_len":len(train_list), "valid_len":len(valid_list)})
        run.log_artifact(artifact)

    train_transforms, val_transforms, easy_unlabel_transforms, hard_unlabel_transforms = call_trans_function_for_semi_easy_hard(config)

    # easy_train_loader = call_dataloader(info, config, train_list, easy_train_transforms, progress=True, mode="train")
    train_loader = call_dataloader(info, config, train_list, train_transforms, progress=True, mode="train")
    valid_loader = call_dataloader(info, config, valid_list, val_transforms, progress=True, mode="valid")
    easy_un_loader = call_dataloader(info, config, unlabel_list, easy_unlabel_transforms, progress=True, mode="train")
    hard_un_loader = call_dataloader(info, config, unlabel_list, hard_unlabel_transforms, progress=True, mode="train")

    # Initialize model, optimizer with teacher model 
    model = call_model(info, config)
    ema_model = call_model(info, config) # teacher model
    ema_model.load_state_dict(torch.load(os.path.join(info["Teacher_pretrained_path"]))["model_state_dict"])
    model.to("cuda")
    ema_model.to("cuda")
    # model = nn.DataParallel(model)

    # ema parameter가 더 이상 loss.backward()에 영향을 받지 않는다.
    for param in ema_model.parameters():
        param.detach_()

    if logging:
        wandb.watch(model, log="all")

    optimizer = call_optimizer(config, model)

    if config["LOAD_MODEL"]:
        check_point = torch.load(os.path.join(info["LOGDIR"], f"model_best.pth"))
        try:
            model.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        except:
            model.load_state_dict(check_point)

    best_loss = 1.
    global_step = 0
    dice_val_best = 0.0
    


    while global_step < config["MAX_ITERATIONS"]:
        global_step, dice_val_best, train_loss, train_dice, unlabel_loss, unlabel_dice = train_for_semi_easy_hard( 
            info, config, global_step, dice_val_best, 
            model, ema_model, optimizer,
            train_loader, easy_un_loader, hard_un_loader, valid_loader,
            logging , deep_supervision=info["Deep_Supervision"]
        )   
        if logging:
            wandb.log({
                'train_loss': train_loss,
                'train_dice': train_dice,
                'unlabel_loss': unlabel_loss,
                'unlabel_dice': unlabel_dice,
            }) 
    if logging:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(
            os.path.join(info["LOGDIR"], f"model_best.pth"), 
            name=f'model/{config["MODEL_NAME"]}')
        run.log_artifact(artifact)
    return 

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('-trainer', default='Neurips_Cellseg_semi_for_easy_hard') 
    args = parser.parse_args()
    info, config = call_config(args.trainer)
    warnings.filterwarnings("ignore")

    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, logging=True)


