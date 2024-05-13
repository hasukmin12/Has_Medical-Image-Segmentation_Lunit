import os

save_name = 'CASwin_DiceFocal_deep_dense_b8_tanh_sdf+sup'

info = {
    "TARGET_NAME"   : "Neurips_Cellseg",
    "VERSION"       : 1,
    "FOLD_for_CrossValidation" : False, # False # 5
    "ROOT"          : "/home/sukmin/datasets/Task150_NeurIPS_Cellseg",
    "CHANNEL_IN"    : 3,
    "CHANNEL_OUT"   : 3,
    "CLASS_NAMES"   : {1:'interior',2:'edge'},
    "NUM_CLASS"     : 2,  

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "Neurips_Cellseg_monai2",
    "VISUAL_AXIS"   : 1, # 1 or 2 or 3
    #### ray
    "GPUS"          : "0",
    "MEM_CACHE"     : 0,
    "VALID_GPU"     : True,
    "Deep_Supervision" : True,
}

info["LOGDIR"] = os.path.join(f'/vast/AI_team/sukmin/Results_monai_{info["TARGET_NAME"]}', save_name)
if os.path.isdir(info["LOGDIR"])==False:
    os.makedirs(info["LOGDIR"])
info["NUM_GPUS"] = len(info["GPUS"].split(','))
info["WORKERS"] = 4 # 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

config = {
    "save_name"     : save_name,
    "LOSS_NAME"     : "DiceFocal",
    "BATCH_SIZE"    : 8,
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
    "MODEL_NAME"    : "caswin_deep_dense_tanh",
    "LOAD_MODEL"    : False,
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],
    # "FOLD"          : info["FOLD"],     
}