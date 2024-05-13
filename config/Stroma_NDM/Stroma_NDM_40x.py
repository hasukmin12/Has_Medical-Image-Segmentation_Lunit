import os

save_name = 'CacoX_b32_40x_e400'

info = {
    "TARGET_NAME"   : "Stroma_NDM",
    "VERSION"       : 1,
    "FOLD_for_CrossValidation" : False, # False # 5
    "ROOT"          : "/vast/AI_team/sukmin/datasets/Task502_Stroma_NDM_prep_40x",
    "CHANNEL_IN"    : 3,
    "CHANNEL_OUT"   : 7,
    "CLASS_NAMES"   : {1:'Stroma', 2:'N-epithelium', 3:'D-epithelium', 4:"M-epithelium", 5:"NET", 6:'others'}, # 0: "white space"
    "NUM_CLASS"     : 6,  

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "Stroma_NDM",
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
info["WORKERS"] = 32 # 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

config = {
    "LOAD_MODEL"    : False,
    "save_name"     : save_name,
    "LOSS_NAME"     : "DiceCE",
    "BATCH_SIZE"    : 128,
    # "BATCH_SIZE_PosNeg" : 4,
    "TRANSFORM"     : 1,
    "SPACING"       : None, # [1.94, 1.94, 3],
    "INPUT_SHAPE"   : [256,256], # [512,512],
    # "DROPOUT"       : 0.1,
    "CONTRAST"      : [0,2000], # [-150,300],
    "FAST"          : False,
    # http://esignal.co.kr/ai-ml-dl/?board_name=ai_ml_dl&search_field=fn_title&order_by=fn_pid&order_type=asc&board_page=1&list_type=list&vid=15
    # s * b = n * e
    # s * 128 = 1881 * 280
    "MAX_ITERATIONS": 2939*2, # (=128 batch, 200 epoch)               # 4114, # (= 128 batch, 280 epoch)    88147 (= 128 batch, 5998 epoch)           
    "EVAL_NUM"      : 293, # (=128 batch, 20 epoch) 
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : 'CacoX', # 'DL_ConvNext_large', # 'DL_se_resnet152', # 'DL_resnext101', # 'DL_dpn131', # "DL_se_resnext101", # DL_regnetx_320  # "MEDIAR_pretrain",  # unet_resnet101_pretrain # 'DeepLabV3Plus_resnet101_pretrain', # "CA_Swin_deep_dense",
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 5e-04, # 1e-04, # 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],    
}
