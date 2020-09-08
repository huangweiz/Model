# _*_ coding:utf-8 _*_
# 开发团队：huang
# 开发时间：2020/9/3 11:42
# 文件名称：get_info.py
# 开发工具：PyCharm

"""
    划分数据集，将分组后的数据集名字和xml解析出来的信息分别写入相应的txt文件中
"""
import os
import random
import xml.etree.ElementTree as ET

voc_imagesets = './dataset/ImageSets'
voc_anno = './dataset/Annotations'


def gen_txt():
    """
    生成四个txt文件
    :return:
    """

    xml_filenames = os.listdir(voc_anno)
    # xml文件数目
    num_xmls = len(xml_filenames)

    # 数据集划分比例
    trainval_percent = 1.0
    val_percent = 0.2
    test_percent = 0.2

    trainval_num = int(num_xmls * trainval_percent)
    val_num = int(trainval_num * val_percent)
    train_num = trainval_num - val_num
    test_num = int(num_xmls * test_percent)

    print("训练集图片数量：", trainval_num)
    print("验证集图片数量：", val_num)
    print("训练集图片数量：", train_num)
    print("测试集图片数量：", test_num)

    # 打开存储文件
    ftrainval = open(os.path.join(voc_imagesets, 'Main/trainval.txt'), 'w')
    ftest = open(os.path.join(voc_imagesets, 'Main/test.txt'), 'w')
    ftrain = open(os.path.join(voc_imagesets, 'Main/train.txt'), 'w')
    fval = open(os.path.join(voc_imagesets, 'Main/val.txt'), 'w')

    # 随机排序，再将信息存入相应文件中
    trainval = random.sample(range(num_xmls), trainval_num)
    train = random.sample(trainval, train_num)
    test = random.sample(range(num_xmls), test_num)

    for i in range(num_xmls):
        name = xml_filenames[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)

        if i in test:
            ftest.write(name)

    ftrainval.close()
    ftest.close()
    ftrain.close()
    fval.close()


gen_txt()

classnames = open('./data/classes/voc.names').readlines()
voc_class = []

for classname in classnames:
    classname = classname.replace('\n', '')
    voc_class.append(classname)

print("数据集中的类别：", voc_class)

# 将与图片对应的xml中的信息写入文件中
sets = ['train', 'val', 'test']


def convert_annotation(image_name, image_infos):
    xml_path = os.path.join(voc_anno, '%s.xml' % image_name)
    xml_file = open(xml_path)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in voc_class:
            continue
        cls_id = voc_class.index(cls)
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text
        b = (int(xmin), int(ymin), int(xmax), int(ymax))
        image_infos.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


for image_set in sets:
    image_names_path = os.path.join(voc_imagesets, 'Main/%s.txt' % image_set)
    image_names = open(image_names_path).read().strip().split()

    image_infos_path = os.path.join(voc_imagesets, 'Info/%s.txt' % image_set)
    image_infos = open(image_infos_path, 'w')

    for image_name in image_names:
        image_infos.write('./dataset/JPEGImages/%s.jpg' % image_name)
        convert_annotation(image_name, image_infos)
        image_infos.write('\n')

    image_infos.close()
