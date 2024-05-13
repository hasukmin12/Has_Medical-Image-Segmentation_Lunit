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
from core.monai_sliding_window_inference_for_DTC import sliding_window_inference_for_tanh

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
from core.sdf_utils import *
from core.loss.for_semi_loss import *
from core.sdf_utils import *
from core.ramps import *
from torch.nn import BCEWithLogitsLoss, MSELoss

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * sigmoid_rampup(epoch, 40.0)


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
            # with torch.no_grad():
            #     gt_dis = compute_sdf(val_labels[:config["BATCH_SIZE"], ...].cpu().numpy(), val_inputs[:config["BATCH_SIZE"], ...].shape)
            #     gt_dis = torch.from_numpy(gt_dis).float().cuda()


            val_outputs = sliding_window_inference_for_tanh(val_inputs, config["INPUT_SHAPE"], 4 , model) # , device='cuda', sw_device='cuda')
            # val_outputs = val_outputs[0]
            
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )

            if info["Deep_Supervision"] == True: 
                dice_class.append(dice_metric(val_outputs[0]>=threshold, val_labels)[0])

                confusion = confusion_matrix(val_outputs[0]>=threshold, val_labels)[0]
                mr_class.append([
                    calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])
                fo_class.append([
                    calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])


            else:
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
        if info["Deep_Supervision"]==True:
            # print(dice_val)
            # print(dice_val.item())
            wandb.log({'valid_dice': dice_val.item(), 'valid_miss rate': miss_val.item(),
            'valid_image': log_image_table(info, val_inputs[0].cpu(),
                                            val_labels[0].cpu(), val_outputs[0][0].cpu())})




        if info["Deep_Supervision"]==False:
        #     # print(dice_val)
        #     # print(dice_val.item())
        #     wandb.log({'valid_dice': dice_val.item(), 'valid_miss rate': miss_val.item(),
        #     'valid_image': log_image_table(info, val_inputs[0].cpu(),
        #                                     val_labels[0].cpu(), val_outputs[0].cpu())})                            
        # # })

            wandb.log(dice_dict)
            # wandb.log(miss_dict)
            # wandb.log(false_dict)        
    return dice_val



def train(info, config, global_step, dice_val_best, model, optimizer, train_loader, valid_loader, logging=False, deep_supervision=True): 
    # print(model)
    loss_function = call_loss(loss_mode=config["LOSS_NAME"], sigmoid=True, config=config)
    dice_loss_f = call_loss(loss_mode='dice', sigmoid=True)
    dicece_loss = call_loss(loss_mode='dicece', sigmoid=True)
    ce_loss_f = call_loss(loss_mode='ce', sigmoid=True)
    mse_loss = MSELoss()

    model.train()

    step = 0
    epoch_loss, epoch_dice_sup, epoch_dice_sdf = 0., 0., 0.
    epoch_consistency = 0.

    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    print(epoch_iterator)
    for step, batch in enumerate(epoch_iterator):
        step += 1

        x, y1 = batch["image"].to("cuda"), batch["label"].to("cuda") # .long()
        # x, y = batch["image"].to("cuda", non_blocking=True), batch["label"].to("cuda", non_blocking=True).long()
        # x = x.squeeze(dim=4)
        # x = x.view([-1, 1, 196, 196, 1])
        logit_map, logit_map_tanh = model(x)

        # 라벨이 1,2,3 되어있는걸 3채널로 분할 + 각각 1씩 넣어준다
        y = one_hot(
                y1, config["CHANNEL_OUT"]
            )  # (b,cls,256,256)


        loss = 0
        loss_sup, dice_sup = 0, 0
        loss_sdf, dice_sdf = 0, 0
        consistency_loss = 0

        if deep_supervision == True:

            # loss for supervised
            for ds in logit_map:
                ce_loss_sup = ce_loss_f(ds, y)
                dice_loss_sup = dice_loss_f(ds, y)
                loss_sup += ce_loss_sup + dice_loss_sup
                dice_sup += 1 - dice_loss_sup

            loss_sup /= len(logit_map)
            dice_sup /= len(logit_map)



            # sdf loss calcuate
            for ds in logit_map_tanh:
                with torch.no_grad():
                    gt_dis = compute_sdf(y[:config["BATCH_SIZE"], ...].cpu().numpy(), ds[:config["BATCH_SIZE"], ...].shape)
                    gt_dis = torch.from_numpy(gt_dis).float().cuda()

                mse_loss_sdf = mse_loss(ds, gt_dis) # tanh 포맷에서의 비교 loss # 본 논문에선 mse loss 활용함
                # dice_loss_sdf = dice_loss_f(ds, gt_dis)
                loss_sdf += mse_loss_sdf 
                # dice_sdf += 1 - dice_loss_sdf

            loss_sdf /= len(logit_map_tanh)
            # dice_sdf /= len(logit_map_tanh)



            # consistency loss
            for i in range(len(logit_map_tanh)):
                dis_to_mask = torch.sigmoid(-1500*logit_map_tanh[i])
                outputs_soft = torch.sigmoid(logit_map[i])
                c_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
                consistency_loss += c_loss
            
            consistency_loss /= len(logit_map_tanh)



             # 최종 loss 조합
            beta = 0.3 # https://github.com/HiLab-git/DTC에서 0.3으로 지정함 ㅎㅎ
            consistency_weight = get_current_consistency_weight(step//150)
            loss = loss_sup + (loss_sdf * beta) + (consistency_weight * consistency_loss)
            






        else:
            print("this is not updated")
            # with torch.no_grad():
            #     gt_dis = compute_sdf(y[:config["BATCH_SIZE"], ...].cpu().numpy(), logit_map_tanh[:config["BATCH_SIZE"], ...].shape)
            #     gt_dis = torch.from_numpy(gt_dis).float().cuda()
            # # loss = loss_function(logit_map, y)
            # ce_loss = ce_loss_f(logit_map, y)
            # dice_loss = dice_loss_f(logit_map, y)
            # dice = 1 - dice_loss
            # loss = ce_loss + dice_loss




        epoch_loss += loss.item()
        epoch_dice_sup += dice_sup.item()
        # epoch_dice_sdf += dice_sdf.item()
        epoch_consistency += consistency_loss.item()
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step+1, config["MAX_ITERATIONS"], loss)
        )
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
    return global_step, dice_val_best, epoch_loss / step, epoch_dice_sup / step, epoch_consistency / step
