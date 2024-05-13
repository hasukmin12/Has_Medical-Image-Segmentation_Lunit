import os

save_name = 'CAD_DL_ConvNext_large_b8_bbbaseline_DiceCE_GF_0.3_TC_0.3_r1024_223' # CAD_DL_se_resnet152_b16_baseline' # 'DL_eff-b8_b4' # 'DL_resnext101_b16' # 'DL_se_resnext101_b16' # 'MEDIAR_pretrain_b8'  # DL_regnety_320_b16  # DeepLabV3Plus_resnet101_pretrain_b32 # unet_resnet101_pretrain_b32_e4k

info = {
    "TARGET_NAME"   : "PAIP_2023",
    "VERSION"       : 1,
    "FOLD_for_CrossValidation" : False, # False # 5
    "ROOT"          : "/vast/AI_team/sukmin/datasets/Task301_PAIP_norm",
    "CHANNEL_IN"    : 3,
    "CHANNEL_OUT"   : 3,
    "CLASS_NAMES"   : {1:'Tumor', 2:'non_tumor'}, # 0: "white space"
    "NUM_CLASS"     : 2,  

    #### wandb
    "ENTITY"        : "hasukmin12",
    "PROJ_NAME"     : "PAIP_2023",
    "VISUAL_AXIS"   : 0, # 1 or 2 or 3

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
info["WORKERS"] = 16 # 4*info["NUM_GPUS"] if info["MEM_CACHE"]>0 else 8*info["NUM_GPUS"]

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
    "MAX_ITERATIONS": 2520*4, # 2940, # 4200,             # 2520, # (= 4batch, 240 epochs)          # 2940, # (= 4batch, 280 epochs)          # 3150 (= 4batch, 300 epochs)            # 4200 (= 4batch, 400 epochs)            8400 (= 4batch, 800 epochs)                    # 10500, # (= 1750 epochs) # 7875, # 4200*4, # 5250, # 4200, # 10000, # 14500, # 20000, # 논문처럼 하려면 800 epoch = 14400 step으로 해야함 -> s*4(b) = 72(n)*800(e) # 여기 참고 : http://esignal.co.kr/ai-ml-dl/?board_name=ai_ml_dl&search_field=fn_title&order_by=fn_pid&order_type=asc&board_page=1&list_type=list&vid=15
    # "MAX_EPOCHS" : 400,
    "EVAL_NUM"      : 50, # 105, # 10, # 105, # (= 4batch, 10 epochs)  # 60, # 6, # 500, #500, 
    "SAMPLES"       : 2,
    "SEEDS"         : 12321,
    "MODEL_NAME"    : 'CAD_DL_ConvNext_large_GF', # 'DL_ConvNext_large', # 'DL_se_resnet152', # 'DL_resnext101', # 'DL_dpn131', # "DL_se_resnext101", # DL_regnetx_320  # "MEDIAR_pretrain",  # unet_resnet101_pretrain # 'DeepLabV3Plus_resnet101_pretrain', # "CA_Swin_deep_dense",
    "OPTIM_NAME"    : "AdamW",
    "LR_INIT"       : 5e-04, # 1e-04, # 5e-04,
    "LR_DECAY"      : 1e-05,
    "MOMENTUM"      : 0.9,
    "CHANNEL_IN"    : info["CHANNEL_IN"], 
    "CHANNEL_OUT"   : info["CHANNEL_OUT"],    
}

