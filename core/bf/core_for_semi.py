import os, sys, glob
sys.path.append('../')
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm

from monai.data import *
from monai.metrics import *
from monai.transforms import Activations
from monai.inferers import sliding_window_inference
from monai.losses.contrastive import ContrastiveLoss
import itertools
from core.utils import *   
from core.call_loss import *
from core.call_model import *
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
import monai
from monai.networks import one_hot
from torch.utils.data.dataset import ConcatDataset
from core.ramps import *
import torch.nn.functional as F
# from core.loss.added_from_FUSSNet import *
# from core.loss.sei_cbc_loss import *
from core.loss.for_semi_loss import *



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 200.0)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def validation(info, config, valid_loader, model, logging=False, threshold=0.5):  
    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!  
    dice_metric = DiceMetric(include_background=False, reduction='none')
    confusion_matrix = ConfusionMatrixMetric(include_background=False, reduction='none')

    epoch_iterator_val = tqdm(
        valid_loader, desc="Validate (X / X Steps)", dynamic_ncols=True
    )
    dice_class, mr_class, fo_class = [], [], []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            step += 1
            
            val_inputs, val_labels = batch["image"].to("cuda"), batch["label"].to("cuda") # , non_blocking=True).long()

            val_labels = one_hot(val_labels, config["CHANNEL_OUT"])
            val_outputs = sliding_window_inference(val_inputs, config["INPUT_SHAPE"], 4 , model) # , device='cuda', sw_device='cuda')
            val_outputs = val_outputs[0]
            
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )
            dice_class.append(dice_metric(val_outputs>=threshold, val_labels)[0])

            confusion = confusion_matrix(val_outputs>=threshold, val_labels)[0]
            mr_class.append([
                calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
            ])
            fo_class.append([
                calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
            ])
            # torch.cuda.empty_cache()
        dice_dict, dice_val = calc_mean_class(info, dice_class, 'valid_dice')
        miss_dict, miss_val = calc_mean_class(info, mr_class, 'valid_miss rate')
        # false_dict, false_val = calc_mean_class(info, fo_class, 'valid_false alarm')
        if logging:
            # print(dice_val)
            # print(dice_val.item())
            wandb.log({'valid_dice': dice_val.item(), 'valid_miss rate': miss_val.item()})

            #'valid_false alarm': false_val})
            # 'valid_image': log_image_table(info, val_inputs[0].cpu(),
            #                                 val_labels[0].cpu(),val_outputs[0].cpu())
        # })

            wandb.log(dice_dict)
            # wandb.log(miss_dict)
            # wandb.log(false_dict)        
    return dice_val



def train_for_semi(info, config, global_step, dice_val_best, 
    model, ema_model, optimizer, 
    train_loader, un_loader, valid_loader, 
    logging=False, deep_supervision=True): 


    loss_function = call_loss(loss_mode=config["LOSS_NAME"], sigmoid=True, config=config)
    dice_loss = call_loss(loss_mode='dice', sigmoid=True)
    ce_loss = call_loss(loss_mode='ce', sigmoid=False)
    # KL_loss = torch.nn.KLDivLoss(reduction='batchmean')
 
    # concat dataloader for semi
    zip_loader = zip(train_loader, un_loader)

    model.train()
    ema_model.train()
    # teacher_model.eval()

    steps = 0
    epoch_loss, epoch_dice = 0., 0.
    un_epoch_loss, un_epoch_dice = 0., 0.
    unc_epoch_loss, unc_epoch_dice = 0., 0.

    loader_num = len(train_loader) + len(un_loader)
    weight_scheduler = ConsistencyWeight(loader_num)

    epoch_iterator = tqdm(
        zip_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for steps, (batch, un_batch) in enumerate(epoch_iterator):
        steps += 1

        # label이 있는 경우 (hard_batch로 student만 학습시킴)
        x, y = batch["image"].to("cuda"), batch["label"].to("cuda") # .long()
        logit_map = model(x)

        y = one_hot(
                y, config["CHANNEL_OUT"]
            )  # (b,cls,256,256)

        su_loss, dice = 0, 0
        if deep_supervision == True: # deep supervision
            for ds in logit_map:
                su_loss += loss_function(ds, y)
                dice += 1 - dice_loss(ds, y)
            su_loss /= len(logit_map)
            dice /= len(logit_map)
        else:
            su_loss = loss_function(logit_map, y)
            dice = 1 - dice_loss(logit_map, y)

        epoch_loss += su_loss.item()
        epoch_dice += dice.item()



        # label이 없는 경우 (un_batch)
        u_x = un_batch["image"].to("cuda")
        logit_map = model(u_x)
        with torch.no_grad():
            t_logit_map = ema_model(u_x)

        # un_loss, un_dice = 0, 0
        # if deep_supervision == True: # deep supervision
        #     for i in range(len(logit_map)):
        #         un_loss += semi_mse_loss(logit_map[i], t_logit_map[i]).mean() # consistency loss from FUSSNet
        #         # un_dice += 1 - dice_loss(logit_map[i], t_logit_map[i])
        #     un_loss /= len(logit_map)
        #     # un_dice /= len(logit_map)
        #     un_dice = 1 - dice_loss(logit_map[0], t_logit_map[0])
        # else:
        #     un_loss = semi_mse_loss(logit_map, t_logit_map).mean()
        #     un_dice = 1 - dice_loss(logit_map, t_logit_map)


        # un_epoch_loss += un_loss.item()
        # un_epoch_dice += un_dice.item()



        un_loss, un_dice = 0, 0
        if deep_supervision == True: # deep supervision
            for i in range(len(logit_map)):
                un_loss += semi_softmax_dice_loss(logit_map[i], t_logit_map[i]).mean() # consistency loss from FUSSNet # semi_cbc_loss # semi_mse_loss
                # un_dice += 1 - dice_loss(logit_map[i], t_logit_map[i])
            un_loss /= len(logit_map)
            # un_dice /= len(logit_map)
            un_dice = 1 - semi_softmax_dice_loss(logit_map[0], t_logit_map[0])
        else:
            un_loss = semi_softmax_dice_loss(logit_map, t_logit_map).maen()
            un_dice = 1 - semi_softmax_dice_loss(logit_map, t_logit_map)

        un_epoch_loss += un_loss.item()
        un_epoch_dice += un_dice.item()

        # consistency_weight = get_current_consistency_weight(steps // 150)
        consistency_weight = weight_scheduler(steps)


        loss = (su_loss) + (un_loss * consistency_weight) # * 0.5 # + (unc_loss * consistency_weight) * 0.3 
        # loss = (su_loss) + (un_loss * 0.5)
        # uncertain_loss * 0.3 #+ certain_loss*0.5
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        update_ema_variables(model, ema_model, 0.99, global_step= steps) # steps) # val 영향

        epoch_iterator.set_description(
            "Training (%d / %d Steps) (su_loss=%2.5f, un_loss=%2.5f, un_dice=%2.5f, consistency_weight=%2.5f, total_loss=%2.5f)" % (global_step+1, config["MAX_ITERATIONS"], su_loss, un_loss, un_dice, consistency_weight, loss)
        )








        # evaluation
        if (
            global_step % config["EVAL_NUM"] == 0 and global_step != 0
        ) or global_step == config["MAX_ITERATIONS"]:
            
            dice_val = validation(info, config, valid_loader, model, logging)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(info["LOGDIR"], f"model_best.pth"))
                print(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )
            else:
                print(
                    f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )

            # 3k마다 model save
            if global_step % 3000 == 0 and global_step != 0 :
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(info["LOGDIR"], "model_e{0:05d}.pth".format(global_step)))
                print(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )

        global_step += 1


    return global_step, dice_val_best, epoch_loss / steps, epoch_dice / steps , un_epoch_loss / steps, un_epoch_dice / steps
