import os

save_name = 'tiger_DDense'

info = {
    "TARGET_NAME"   : "tiger_model",
    "VERSION"       : 1,
    "FOLD"          : 4,
    "FOLDS"         : 5, 
    "ROOT"          : "/vast/AI_team/sukmin/datasets/Task001_TIGER",
    "CHANNEL_IN"    : 3,
    "CHANNEL_OUT"   : 8,
    "CLASS_NAMES"   : {1:'invasive tumor',2:'tumor-associated stroma',3:'in-situ tumor',4:'healthy glands',5:'necrosis not in-situ', 6:'inflamed stroma', 7:'rest'},  

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "tiger_monai",
    "VISUAL_AXIS"   : 2, # 1 or 2 or 3
    #### ray
    "GPUS"          : "0",
    "MEM_CACHE"     : 0.5,
    "VALID_GPU"     : True,
}

info["LOGDIR"] = os.path.join(f'/vast/AI_team/sukmin/{info["TARGET_NAME"]}', save_name)
if os.path.isdir(info["LOGDIR"])==False:
    os.makedirs(info["LOGDIR"])
info["NUM_GPUS"] = len(info["GPUS"].split(','))
info["WORKERS"] = 1 # 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

config = {
    "save_name"     : save_name,
    "LOSS_NAME"     : "DiceCE",
    "BATCH_SIZE"    : 1,
    "TRANSFORM"     : 1,
    "SPACING"       : None, # [1.94, 1.94, 3],
    "INPUT_SHAPE"   : [192,192],
    "DROPOUT"       : 0.1,
    "CONTRAST"      : [0,2000], # [-150,300],
    "FAST"          : True,
    "MAX_ITERATIONS": 100000,
    "EVAL_NUM"      : 500, # 500,
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : "unet",
    "LOAD_MODEL"    : False,
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    #### DoNotChange! by JEpark
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],
    "FOLD"          : info["FOLD"],     
}