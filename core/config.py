# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 16:09
# @Author  : huangwei
# @File    : config.py
# @Software: PyCharm

from easydict import EasyDict as edict

__C                             = edict()

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

__C.YOLO.CLASSES                = "./data/classes/voc.names"
__C.YOLO.ANCHORS                = "./data/anchors/anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "./dataset/ImageSets/Info/train.txt"
__C.TRAIN.BATCH_SIZE            = 1
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 5
__C.TRAIN.SECOND_STAGE_EPOCHS   = 5
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_test_loss=2.5736.ckpt-10"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "./dataset/ImageSets/Info/test.txt"
__C.TEST.BATCH_SIZE             = 1
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_test_loss=1.2967.ckpt-10"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.1
__C.TEST.IOU_THRESHOLD          = 0.1






