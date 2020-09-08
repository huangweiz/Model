# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 11:13
# @Author  : huangwei
# @File    : demo5.py
# @Software: PyCharm

import os
import shutil

import cv2
from core import utils
from core.config import cfg
import numpy as np
import tensorflow as tf
from core.yolov3 import YOLOV3
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class YoloTest(object):
    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.weight_file = cfg.TEST.WEIGHT_FILE
        self.show_label = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable = tf.placeholder(dtype=tf.bool, name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        # 将 image 转成需要的格式和 大小
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        # 给图片加一个维度，加在最前面
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        # concatenate() 多个数组的拼接
        # reshape(data, newshape)
        # newshape=(-1, nums) 表示生成一个 (rows, cols)的形状，cols = nums, rows的值根据cols的值计算
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        # 筛选出有效，在范围内的框
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)

        # 筛选出一个最合适的框
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self, image_path):
        image = cv2.imread(image_path)

        image_name = image_path.split('\\')[-1].split('/')[-1]
        filename = image_name.split('.')[-2]

        predict_image_path = './output/' + image_name
        predict_result_path = './output/' + filename + '.txt'

        bboxes_pr = self.predict(image)

        image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
        cv2.imwrite(predict_image_path, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())


if __name__ == '__main__':
    image_path = './dataset/JPEGImages/img-316.jpg'

    YoloTest().evaluate(image_path)
