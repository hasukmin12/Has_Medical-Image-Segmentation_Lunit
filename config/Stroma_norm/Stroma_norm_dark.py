import os

save_name = 'ResUNet_b32_norm_dark_prep16' 

info = {
    "TARGET_NAME"   : "Stroma_norm",
    "VERSION"       : 1,
    "FOLD_for_CrossValidation" : False, # False # 5
    "ROOT"          : "/vast/AI_team/sukmin/datasets/Task202_Stroma_prep_dark",
    "CHANNEL_IN"    : 3,
    "CHANNEL_OUT"   : 4,
    "CLASS_NAMES"   : {1:'Stroma', 2:'epithelium', 3:'others'}, # 0: "white space"
    "NUM_CLASS"     : 3,  

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "Stroma",
    "VISUAL_AXIS"   : 1, # 1 or 2 or 3
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
info["WORKERS"] = 8 # 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

config = {
    "save_name"     : save_name,
    "LOSS_NAME"     : "DiceFocal",
    "BATCH_SIZE"    : 2,
    "BATCH_SIZE_PosNeg" : 16,
    "TRANSFORM"     : 1,
    "SPACING"       : None, # [1.94, 1.94, 3],
    "INPUT_SHAPE"   : [512,512],
    "DROPOUT"       : 0.0,
    "CONTRAST"      : [0,2000], # [-150,300],
    "FAST"          : False,
    "MAX_ITERATIONS": 20000,
    "EVAL_NUM"      : 500, # 500,
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : 'unet', # "CA_Swin_deep_dense",
    "LOAD_MODEL"    : False,
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],
    # "FOLD"          : info["FOLD"],     
}