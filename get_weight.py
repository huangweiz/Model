# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 15:55
# @Author  : huangwei
# @File    : get_weight.py
# @Software: PyCharm

"""
    使用 yolov3_coco.ckpt 文件
    生成初始的 权重文件 yolov3_coco_demo.ckpt
    使用 yolov3_coco.ckpt 文件和 anchor box， classnames 构建一个模型，保存到 demo文件中，‘
    用于接下来训练数据使用
"""
import argparse

import tensorflow as tf
from core.config import cfg
from core.yolov3 import YOLOV3

org_weights_path = cfg.YOLO.ORIGINAL_WEIGHT
cur_weights_path = cfg.YOLO.DEMO_WEIGHT
preserve_org_names = ['Conv_6', 'Conv_14', 'Conv_22']
preserve_cur_names = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']

# 实例化一个数据流图作为tensorflow运行环境的默认图
tf.Graph().as_default()

# 用来加载meta文件中的图，以及图上定义的结点参数包括权重配置项等需要训练的参数
# 也包括训练过程中生成的中间参数，所有的参数都是通过graph调用接口
# get_tensor_by_name(name="训练时的参数名称")来获取
load = tf.train.import_meta_graph(org_weights_path + '.meta')

org_weights_mess = []
cur_weights_mess = []

with tf.Session() as sess:
    # 恢复模型
    load.restore(sess, org_weights_path)
    for var in tf.global_variables():
        var_name = var.op.name
        var_name_mess = str(var_name).split('/')

        var_shape = var.shape

        op1 = var_name_mess[-1] not in ['weights', 'gamma', 'beta', 'moving_mean', 'moving_variance']
        op2 = var_name_mess[1] == 'yolo-v3' and (var_name_mess[-2] in preserve_org_names)
        if op1 or op2:
            continue

        org_weights_mess.append([var_name, var_shape])
        print("1=> " + str(var_name).ljust(50), var_shape)

print()

# 清除并重置全局默认图形
tf.reset_default_graph()

tf.Graph().as_default()

# 创建模型
with tf.name_scope('input'):
    # placeholder()函数在神经网络构建graph时在模型中占位，并没有把要输入的数据传入模型只会分配必要的内存
    input_data = tf.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3), name='input_data')
    training = tf.placeholder(dtype=tf.bool, name='trainable')

model = YOLOV3(input_data, training)

for var in tf.global_variables():
    var_name = var.op.name
    var_name_mess = str(var_name).split('/')
    var_shape = var.shape


    if var_name_mess[0] in preserve_cur_names:
        continue

    cur_weights_mess.append([var_name, var_shape])
    print("2=> " + str(var_name).ljust(50), var_shape)

print()

org_weights_num = len(org_weights_mess)
cur_weights_num = len(cur_weights_mess)

if org_weights_num != cur_weights_num:
    raise RuntimeError

print("3=> number of weights that will rename:%d" % cur_weights_num)

cur_to_org_dict = {}
for index in range(org_weights_num):
    org_name, org_shape = org_weights_mess[index]
    cur_name, cur_shape = cur_weights_mess[index]

    if cur_shape != org_shape:
        print(org_weights_mess[index])
        print(cur_weights_mess[index])
        raise RuntimeError
    cur_to_org_dict[cur_name] = org_name
    print("4=> " + str(cur_name).ljust(50) + ' : ' + org_name)

with tf.name_scope('load_save'):
    name_to_var_dict = {var.op.name: var for var in tf.global_variables()}
    restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
    load = tf.train.Saver(restore_dict)
    save = tf.train.Saver(tf.global_variables())
    for var in tf.global_variables():
        print("5=> " + var.op.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("6=> restoring weights from:\t %s" % org_weights_path)
    load.restore(sess, org_weights_path)
    save.save(sess, cur_weights_path)

tf.reset_default_graph()
