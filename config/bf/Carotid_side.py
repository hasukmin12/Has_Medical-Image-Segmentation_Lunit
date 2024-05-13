import os

save_name = 'UNet_b16_2'

info = {
    "TARGET_NAME"   : "Carotid_side",
    "VERSION"       : 1,
    "FOLD"          : 5,
    "FOLDS"         : 6,
    "FOLD_for_CrossValidation" : False, # False # 5
    "ROOT"          : "/home/sukmin/datasets/Task113_Carotid_side", # Task111_Carotid_side
    "CHANNEL_IN"    : 3,
    "CHANNEL_OUT"   : 3,
    "CLASS_NAMES"   : {1:'Longitudinal stenosis area',2:'Longitudinal lumen'},
    "NUM_CLASS"     : 2,  

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "Carotid_side2",
    "VISUAL_AXIS"   : 2, # 1 or 2 or 3
    #### ray
    "GPUS"          : "0",
    "MEM_CACHE"     : 0,
    "VALID_GPU"     : True,
    "Deep_Supervision" : False,
}

info["LOGDIR"] = os.path.join(f'/vast/AI_team/sukmin/Results_monai_{info["TARGET_NAME"]}', save_name)
if os.path.isdir(info["LOGDIR"])==False:
    os.makedirs(info["LOGDIR"])
info["NUM_GPUS"] = len(info["GPUS"].split(','))
info["WORKERS"] = 4 # 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

config = {
    "save_name"     : save_name,
    "LOSS_NAME"     : "DiceCe",
    "BATCH_SIZE"    : 16,
    "TRANSFORM"     : 1,
    "SPACING"       : None, # [1.94, 1.94, 3],
    "INPUT_SHAPE"   : [512,512],
    "DROPOUT"       : 0.1,
    "CONTRAST"      : [0,2000], # [-150,300],
    "FAST"          : False,
    "MAX_ITERATIONS": 40000,
    "EVAL_NUM"      : 500, # 500,
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : "UNet",
    "LOAD_MODEL"    : False,
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 1e-03,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],
    "FOLD"          : info["FOLD"],     
}