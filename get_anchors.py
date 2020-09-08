# _*_ coding:utf-8 _*_
# 开发团队：huang
# 开发时间：2020/9/2 16:23
# 文件名称：get_anchors.py
# 开发工具：PyCharm

"""
    用来生成 anchors box 的大小， cluster_num 为 anchors box 的数量
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

# 图片分组后，各组图片名存储目录
voc_imagesets = './dataset/ImageSets'
# 图片存储目录
voc_images = './dataset/JPEGImages'
# xml文件存储目录
voc_anno = './dataset/Annotations'

xml_filenames = os.listdir(voc_anno)
# xml文件数目
num_xmls = len(xml_filenames)

# 先将所有的图片名放到trainval中
trainval_path = os.path.join(voc_imagesets, 'Main/trainval.txt')

ftrainval = open(trainval_path, 'w')

# 遍历所有xml文件，获取其文件名，存储到 trainval中
for i in range(num_xmls):
    name = xml_filenames[i][:-4] + '\n'
    ftrainval.write(name)

ftrainval.close()

# 给类名分配编号
classnames = open('./data/classes/voc.names').readlines()
names_dict = {}
cnt = 0
for name in classnames:
    name = name.strip()
    names_dict[name] = cnt
    cnt += 1


def parse_xml(xml_name):
    """
    传入xml 文件路径，解析出标识文件中的元素值，存储到xml_info对象中
    xml_info对象的格式为 [image_name, width, height, class_id,
    x_min, y_min, x_max, y_max (, class_id, x_min, y_min, x_max, y_max)]
    :param xml_name:
    :return:
    """
    tree = ET.parse(xml_name)

    # 取出需要的元素值存储到xml_info对象中
    image_name = xml_name.split('/')[-1][:-4]

    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    xml_info = [image_name, width, height]

    for obj in tree.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        # 获得类名编号
        name_id = str(names_dict[name])

        xml_info.extend([name_id, xmin, ymin, xmax, ymax])

    if len(xml_info) > 1:
        return xml_info
    else:
        return None


def gen_text(text_path):
    """
    将xml中的信息保存到txt文件中。文件中每一行的格式为：
    [count, image_path, width, height, class_id, x_min, y_min, x_max, y_max (, class_id, x_min, y_min, x_max, y_max)]
    :param text_path:
    :return:
    """

    count = 0
    f = open(text_path, 'w')

    image_names = open(trainval_path).readlines()
    for image_name in image_names:
        image_name = image_name.strip()
        xml_name = voc_anno + '/' + image_name + '.xml'

        objects = parse_xml(xml_name)

        if objects:
            objects[0] = voc_images + '/' + objects[0] + '.jpg'

            if os.path.exists(objects[0]):
                objects.insert(0, str(count))
                count += 1
                objects = ' '.join(objects) + '\n'
                f.write(objects)

    f.close()


gen_text('./trainval.txt')


def parse_info(train_txt_path, target_size):
    """
    将所有图片转换成同样大小，然后求出被标记方框在转换后的长和宽
    将所有被标记方框的长和宽存储在boxes中，用以后面求聚类
    :param train_txt_path:
    :param target_size:
    :return:
    """
    lines = open(train_txt_path)
    boxes = []

    for line in lines:
        info = line.strip().split(' ')

        img_w = int(info[2])
        img_h = int(info[3])

        info = info[4:]
        num_box = len(info) // 5

        for i in range(num_box):
            x_min, y_min, x_max, y_max = float(info[i * 5 + 1]), float(info[i * 5 + 2]), float(info[i * 5 + 3]), float(
                info[i * 5 + 4])
            width = x_max - x_min
            height = y_max - y_min

            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                boxes.append([width, height])
            else:
                boxes.append([width, height])

    boxes = np.asarray(boxes)
    return boxes


def iou(box, clusters):
    """
    计算box 和 含有 9个 box 的 clusters 的 iou

    计算原理未知
    :param box:
    :param clusters:
    :return:
    """
    # 将clusters[][0] 数组中比box[0] 大的数替换为box[0]
    # 将clusters[][1] 数组中比box[1] 大的数替换为box[1]
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    # 如果 x 中元素全为0
    if np.count_nonzero(x == 0) >= clusters.shape[0] or np.count_nonzero(y == 0) >= clusters.shape[0]:
        raise ValueError("box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)

    return iou_


def kmeans(boxes, cluster_num, dist=np.median):
    """
    将所有长宽值作为一个数组传入
    先随机找到 cluster_num 个box
    计算其他 box 和这9个 box的 iou
    取 iou 最小时的 坐标，即该box 被分到9个类中的某一类
    :param boxes:
    :param cluster_num:
    :param dist:
    :return:
    """
    rows = boxes.shape[0]

    # 未初始化和全为0 的两个数组
    distances = np.empty((rows, cluster_num))
    last_clusters = np.zeros((rows,))  # 用于存储分类结果的数组

    np.random.seed()

    # 从boxes中随机取出 9个没有重复的box 放到 clusters中
    clusters = boxes[np.random.choice(rows, cluster_num, replace=False)]

    while True:
        for row in range(rows):
            # 求出这个box 和clusters中每个box的 iou
            distances[row] = 1 - iou(boxes[row], clusters)

        # 求出每一行最小值的坐标，即 iou最小时，这个box属于 9类中的某一类
        min_index = np.argmin(distances, axis=1)

        # 如果与上一轮分类没有区别，则分类完成退出
        if (last_clusters == min_index).all():
            break

        for cluster in range(cluster_num):
            # boxes[min_index == cluster]  用于取出所有属于第 cluster 类的 box
            # dist(boxes[min_index == cluster], axis=0)  用于计算该类的box 的长宽的中位数
            # 更新 clusters中的值，使得聚类效果更好
            clusters[cluster] = dist(boxes[min_index == cluster], axis=0)

        # 更新分类结果
        last_clusters = min_index

    # 返回 更加贴合的分类框大小
    return clusters


def avg_iou(boxes, clusters):
    """
    计算平均的 iou
    :param boxes:
    :param clusters:
    :return:
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def get_kmeans(boxes, cluster_num):
    """
    得到 anchors box 的大小和 平均的 iou
    :param boxes:
    :param cluster_num:
    :return:
    """
    anchors = kmeans(boxes, cluster_num)
    average_iou = avg_iou(boxes, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, average_iou


if __name__ == '__main__':
    target_size = [608, 608]
    train_txt_path = './trainval.txt'

    boxes = parse_info(train_txt_path, target_size=target_size)

    anchors, average_iou = get_kmeans(boxes, 9)

    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])

    anchor_string = anchor_string[:-2]

    print("anchors are:" + anchor_string)
    print("the average iou is:", average_iou)
