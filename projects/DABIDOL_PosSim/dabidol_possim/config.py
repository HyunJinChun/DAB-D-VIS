# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_dabidol_possim_config(cfg):
    """
    Add config for DAB-IDOL_PosSim.
    """
    cfg.MODEL.IDOL = CN()
    cfg.MODEL.IDOL.NUM_CLASSES = 80

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    cfg.INPUT.COCO_PRETRAIN = False
    cfg.INPUT.PRETRAIN_SAME_CROP = False

    # LOSS
    cfg.MODEL.IDOL.MASK_WEIGHT = 2.0
    cfg.MODEL.IDOL.DICE_WEIGHT = 5.0
    cfg.MODEL.IDOL.GIOU_WEIGHT = 2.0
    cfg.MODEL.IDOL.L1_WEIGHT = 5.0
    cfg.MODEL.IDOL.CLASS_WEIGHT = 2.0
    cfg.MODEL.IDOL.REID_WEIGHT = 2.0
    cfg.MODEL.IDOL.DEEP_SUPERVISION = True
    cfg.MODEL.IDOL.MASK_STRIDE = 4
    cfg.MODEL.IDOL.MATCH_STRIDE = 4
    cfg.MODEL.IDOL.FOCAL_ALPHA = 0.25

    cfg.MODEL.IDOL.SET_COST_CLASS = 2
    cfg.MODEL.IDOL.SET_COST_BOX = 5
    cfg.MODEL.IDOL.SET_COST_GIOU = 2

    # TRANSFORMER
    cfg.MODEL.IDOL.NHEADS = 8
    cfg.MODEL.IDOL.DROPOUT = 0.1
    cfg.MODEL.IDOL.DIM_FEEDFORWARD = 1024  # intermediate size of the feedforward layers in the transformer blocks ## DAB-DETR에서는 2048
    cfg.MODEL.IDOL.ENC_LAYERS = 6
    cfg.MODEL.IDOL.DEC_LAYERS = 6

    cfg.MODEL.IDOL.HIDDEN_DIM = 256
    cfg.MODEL.IDOL.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.IDOL.DEC_N_POINTS = 4
    cfg.MODEL.IDOL.ENC_N_POINTS = 4
    cfg.MODEL.IDOL.NUM_FEATURE_LEVELS = 4  # number of feature levels

    # * DAB-Deformable DETR
    cfg.MODEL.IDOL.USE_DAB = True
    cfg.MODEL.IDOL.PRE_NORM = True  # using pre-norm in the Transformer blocks
    cfg.MODEL.IDOL.NUM_SELECT =300  # the number of predictions selected for evaluation
    cfg.MODEL.IDOL.TRANSFORMER_ACTIVATION = 'prelu'
    cfg.MODEL.IDOL.NUM_PATTERNS = 0  # number of pattern embeddings. See Anchor DETR for more details
    cfg.MODEL.IDOL.RANDOM_REFPOINTS_XY = False  # random init the x,y of anchor boxes and freeze them
    cfg.MODEL.IDOL.TWO_STAGE = False  # using two stage variant for DAB-Deformable-DETR

    # Evaluation
    cfg.MODEL.IDOL.CLIP_STRIDE = 1
    cfg.MODEL.IDOL.MERGE_ON_CPU = True
    cfg.MODEL.IDOL.MULTI_CLS_ON = True
    cfg.MODEL.IDOL.APPLY_CLS_THRES = 0.05

    cfg.MODEL.IDOL.TEMPORAL_SCORE_TYPE = 'mean' # mean or max score for sequence masks during inference,
    cfg.MODEL.IDOL.INFERENCE_SELECT_THRES = 0.1  # 0.05 for ytvis
    cfg.MODEL.IDOL.NMS_PRE =  0.5
    cfg.MODEL.IDOL.ADD_NEW_SCORE = 0.2
    cfg.MODEL.IDOL.INFERENCE_FW = True #frame weight
    cfg.MODEL.IDOL.INFERENCE_TW = True  #temporal weight
    cfg.MODEL.IDOL.MEMORY_LEN = 3
    cfg.MODEL.IDOL.BATCH_INFER_LEN = 10

    cfg.MODEL.IDOL.MATCH_SCORE_THR = 0.5  # f(i,j) > match_score_thr

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # * Cascade in Inference
    cfg.MODEL.IDOL.CASCADE = False

    # * Position Similarity in Inference
    cfg.MODEL.IDOL.POSSIM = False

    ## support Swin backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # find_unused_parameters
    cfg.FIND_UNUSED_PARAMETERS = True