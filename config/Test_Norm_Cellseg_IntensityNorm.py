import os

save_name = 'Test_Norm_Cellseg_IntensityNorm' 

info = {
    "TARGET_NAME"   : "Test_Norm",
    "VERSION"       : 1,
    "FOLD_for_CrossValidation" : False, # False # 5
    "ROOT"          : "/vast/AI_team/sukmin/datasets/Test_Norm/Cellseg/Task004_Test_Norm_Cellseg_IntensityNorm",
    "CHANNEL_IN"    : 3, # RGB = 3
    "CHANNEL_OUT"   : 3, # CLASS_NAMES 길이 + 1
    "CLASS_NAMES"   : {1:'Cell', 2:'Cell Boundary'}, # 0: "white space"
    "NUM_CLASS"     : 2,  

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "Norm_Test",
    "VISUAL_AXIS"   : 1, # 1 or 2 or 3

    #### ray
    "GPUS"          : "0",
    "MEM_CACHE"     : 0,
    "VALID_GPU"     : True,
    "Deep_Supervision" : False,
}

info["LOGDIR"] = os.path.join(f'/vast/AI_team/sukmin/Test_Norm/Cellseg/Results_monai_{info["TARGET_NAME"]}', save_name)
if os.path.isdir(info["LOGDIR"])==False:
    os.makedirs(info["LOGDIR"])
info["NUM_GPUS"] = len(info["GPUS"].split(','))
info["WORKERS"] = 8 # 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

config = {
    "LOAD_MODEL"    : False,
    "save_name"     : save_name,
    "LOSS_NAME"     : "DiceCE",
    "BATCH_SIZE"    : 8,
    # "BATCH_SIZE_PosNeg" : 4,
    "TRANSFORM"     : 1,
    "SPACING"       : None, # [1.94, 1.94, 3],
    "INPUT_SHAPE"   : [1024, 1024], # [512,512],
    # "DROPOUT"       : 0.1,
    "CONTRAST"      : [0,2000], # [-150,300],
    "FAST"          : False,

    # s * b = n * e
    # s * 32 = 10074 * 200
    # s * 32*4 = 3455 * 300
    # s * 4 = 1000 * 200
    "MAX_ITERATIONS": 25000, # (= 16batch, 200 epochs)   # 여기 참고 : http://esignal.co.kr/ai-ml-dl/?board_name=ai_ml_dl&search_field=fn_title&order_by=fn_pid&order_type=asc&board_page=1&list_type=list&vid=15
    # "MAX_EPOCHS" : 400,
    "EVAL_NUM"      : 1250, ## (= 16batch, 10 epochs)  # 60, # 6, # 500, #500, 
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : 'DeepLabV3Plus_resnet101_pretrain', # 'DL_ConvNext_large', # 'DL_se_resnet152', # 'DL_resnext101', # 'DL_dpn131', # "DL_se_resnext101", # DL_regnetx_320  # "MEDIAR_pretrain",  # unet_resnet101_pretrain # 'DeepLabV3Plus_resnet101_pretrain', # "CA_Swin_deep_dense",
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 5e-04, # 1e-04, # 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],    
}

