import os

save_name = 'CASwin_DiceFocal_deep_dense_dice' # _plus_e50k

info = {
    "TARGET_NAME"   : "Neurips_Cellseg_semi_af_chall",
    "VERSION"       : 1,
    "FOLD"          : 4,
    "FOLDS"         : 10, 
    "ROOT"          : "/home/sukmin/datasets/Task080_NeurIPS_Cellseg",
    "CHANNEL_IN"    : 3,
    "CHANNEL_OUT"   : 3,
    "CLASS_NAMES"   : {1:'interior',2:'edge'},
    "NUM_CLASS"     : 2,  

    # for semi-supervised
    "ROOT_unlabeled": "/home/sukmin/datasets/Task021_NeurIPS_Cellseg_unlabeled_patches/imagesTr",
    "Teacher_pretrained_path" : "/vast/AI_team/sukmin/Results_monai_Neurips_Cellseg/CASwin_DiceFocal_deep_dense_b16/model_e20000.pth",
    "Continue_training" : False,
    # "Continue_training": "/vast/AI_team/sukmin/Results_monai_Neurips_Cellseg_semi/Final_ST_CASwin_DiceFocal_deep_dense_dice_e60k_plus_e20k_diff_fold/model_best.pth",

    # "/home/sukmin/NeurIPS-CellSeg/baseline/work_dir/swinunetr_3class/best_Dice_model.pth",
    # "/vast/AI_team/sukmin/Results_monai_Neurips_Cellseg/UNet_DiceCe/model_best.pth",

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "Neurips_Cellseg_monai",
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
    "TRANSFORM"     : 2,
    "SPACING"       : None, # [1.94, 1.94, 3],
    "INPUT_SHAPE"   : [512,512],
    "DROPOUT"       : 0.1,
    "CONTRAST"      : [0,2000], # [-150,300],
    "FAST"          : False,
    "MAX_ITERATIONS": 60000,
    "EVAL_NUM"      : 500, # 500,
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : "caswin_deep_dense",
    "LOAD_MODEL"    : False,
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 1e-03,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],
    "FOLD"          : info["FOLD"],     
}