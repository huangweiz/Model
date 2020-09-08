# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 9:44
# @Author  : huangwei
# @File    : demo6.py
# @Software: PyCharm

import argparse
import glob
import json
import operator
import os
import shutil
import sys

MINOVERLAP = 0.5

# 参数值
parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help='no animation is shown.', action='store_true')  # 无动画
parser.add_argument('-np', '--no-plot', help='no plot is show.', action='store_true')
parser.add_argument('-q', '--quiet', help='minimalistic console output.', action='store_true')
parser.add_argument('-i', '--ignore', nargs='+', type=str, help='ignore a list of classes.')
parser.add_argument('--set-class-iou', nargs='+', type=str, help='set iou for a specific class.')

args = parser.parse_args()
args.no_animation = True

# 如果args.ignore 是  None，将其替换为空列表
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# if there are no images then no animation can be shown
img_path = 'images'
if os.path.exists(img_path):
    for dirpath, dirnames, files in os.walk(img_path):
        if not files:
            args.no_animation = True
else:
    args.no_animation = True

# 如果没有选择--na 则导入 opencv
show_animation = False
if not args.no_animation:
    try:
        import cv2

        show_animation = True
    except ImportError:
        print("please install opencv-python!")
        args.no_animation = True

# 如果没有选择 --no-plot 则导入matplotlib
draw_plot = False
if not args.no_plot:
    try:
        import matplotlib.pyplot as plt

        draw_plot = True
    except ImportError:
        print("please install matplotlib!")
        args.no_plot = True


def error(msg):
    """
        print error message and exit
    """
    print(msg)
    sys.exit(0)


def is_float(value):
    """
        判断是否为 0 到 1 之间的浮点数
    """
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


def adjust_axes(r, t, fig, axes):
    """
        Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    """
        Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def voc_ap(rec, prec):
    """
        根据给的recall 和 array 计算 AP
        1. 根据精度单调下降计算 precision/recall 曲线
        2. 使用数值积分计算在曲线下的面积作为 AP
    """

    rec.insert(0, 0.0)  # 插入起始点
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def file_to_list(path):
    """
    convert file lines to a list
    :param path:
    :return:
    """
    with open(path) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    return content


def draw_text_in_image(img, text, pos, color, line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    text_corner = pos
    cv2.putText(img, text, text_corner, font, fontScale, color, lineType)
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, (line_width + text_width)


# 创建 "tmp_files/" 和 "results/" 文件夹
tmp_files_path = "tmp_files"
if not os.path.exists(tmp_files_path):
    os.makedirs(tmp_files_path)
results_files_path = "results"
if os.path.exists(results_files_path):
    shutil.rmtree(results_files_path)
os.makedirs(results_files_path)

if draw_plot:
    os.makedirs(results_files_path + "/classes")

if show_animation:
    os.makedirs(results_files_path + "/images")
    os.makedirs(results_files_path + "/images/single_predictions")

# 存储每个类数量的字典
class_counter = {}

"""
    将ground-truth文件夹中的数据存储到 .json 文件中
"""
ground_truth_files_list = glob.glob('ground-truth/*.txt')
if len(ground_truth_files_list) == 0:
    error("error: no ground-truth files found!")

ground_truth_files_list.sort()

for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))

    # 检查是否有一个预测值文件存在
    if not os.path.exists('predicted/' + file_id + '.txt'):
        error_msg = "file not found: predicted/" + file_id + ".txt\n"
        error_msg += "this is a null space !"
        error(error_msg)

    lines_list = file_to_list(txt_file)

    bounding_boxes = []

    for line in lines_list:
        try:
            class_name, left, top, right, bottom = line.split()
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format.\n"
            error_msg += " Expected: <class_name> <left> <top> <right> <bottom> \n"
            error_msg += " Received: " + line
            error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
            error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
            error(error_msg)

        # 检查 class 是否在 ignore list中
        if class_name in args.ignore:
            continue

        bbox = left + " " + top + " " + right + " " + bottom
        bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

        if class_name in class_counter:
            class_counter[class_name] += 1
        else:
            class_counter[class_name] = 1

    # 将 bounding boxes 信息存入 .json文件中
    with open(tmp_files_path + "/" + file_id + "_ground_truth.json", "w") as outfile:
        json.dump(bounding_boxes, outfile)

# 存储类名的列表
class_names = list(class_counter.keys())

# 将 class 按字母排序
class_names = sorted(class_names)
class_num = len(class_names)

# 为 classes 设置了 iou
if specific_iou_flagged:
    class_iou = args.set_class_iou
    n_args = len(class_iou)
    error_msg = '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'

    if n_args % 2 != 0:
        error('Error, missing arguments. Flag usage:' + error_msg)

    iou_classes = class_iou[::2]
    iou_list = class_iou[1::2]

    if len(iou_classes) != len(iou_list):
        error('Error, missing arguments. Flag usage:' + error_msg)

    for tmp_class in iou_classes:
        if tmp_class not in class_names:
            error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)

    for num in iou_list:
        if not is_float(num):
            error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
    将 predicted 文件夹中的数据存储到 .json 文件中
"""
predicted_files_list = glob.glob('predicted/*.txt')
predicted_files_list.sort()

for class_index, class_name in enumerate(class_names):
    bounding_boxes = []
    for txt_file in predicted_files_list:
        file_id = txt_file.split('.txt', 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        if class_index == 0:
            if not os.path.exists('ground-truth/' + file_id + '.txt'):
                error_msg = "file not found: predicted/" + file_id + ".txt\n"
                error_msg += "this is a null space !"

        lines = file_to_list(txt_file)
        for line in lines:
            try:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                error(error_msg)
            if tmp_class_name == class_name:
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
    bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)

    with open(tmp_files_path + '/' + class_name + "_predictions.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

# 计算每一个 class 的 AP
sum_ap = 0.0
ap_dictionary = {}

# 将结果存储到文件中
with open(results_files_path + '/results.txt', 'w') as results_file:
    results_file.write("# AP and precision/recall per class\n")
    result_dictionary = {}

    for class_index, class_name in enumerate(class_names):
        result_dictionary[class_name] = 0

        # 加载 预测值
        predictions_file = tmp_files_path + '/' + class_name + '_predictions.json'
        predictions_data = json.load(open(predictions_file))

        data_num = len(predictions_data)

        # 创建两个全为0 的数组
        tp = [0] * data_num
        fp = [0] * data_num

        # print(predictions_data)
        for idx, prediction in enumerate(predictions_data):
            file_id = prediction['file_id']
            if show_animation:
                # 找到 image 去显示
                print(img_path)
                ground_truth_img = glob.glob1(img_path, file_id + '.*')
                print(ground_truth_img)

                if len(ground_truth_img) == 0:
                    error("error. image not found with id:" + file_id)
                elif len(ground_truth_img) > 1:
                    error("error. multiple images with id:" + file_id)
                else:
                    img = cv2.imread(img_path + '/' + ground_truth_img[0])

                    """
                        space
                    """

            # 根据file_id 加载 ground_truth值
            tmp_file = tmp_files_path + '/' + file_id + '_ground_truth.json'
            ground_truth_data = json.load(open(tmp_file))

            # iou 的最大值
            ioumax = -1

            # 匹配的 data
            match = {}

            # 加载预测的 bounding-box
            pred_box = [float(x) for x in prediction['bbox'].split()]

            for obj in ground_truth_data:
                # 匹配ground-truth 中 classname
                if obj['class_name'] == class_name:
                    # 加载匹配的 bounding-box
                    ground_box = [float(x) for x in obj['bbox'].split()]

                    # 两个 bounging-box 的边界值
                    box_i = [max(pred_box[0], ground_box[0]), max(pred_box[1], ground_box[1]),
                             min(pred_box[2], ground_box[2]), min(pred_box[3], ground_box[3])]

                    # 计算两个box 相交的矩形的长宽
                    iw = box_i[2] - box_i[0] + 1
                    ih = box_i[3] - box_i[1] + 1

                    if iw > 0 and ih > 0:
                        # 计算 IOU
                        # 两个矩形叠加之后的面积
                        ua = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1) \
                             + (ground_box[2] - ground_box[0] + 1) * (ground_box[3] - ground_box[1] + 1) - iw * ih

                        iou = iw * ih / ua
                        if iou > ioumax:
                            ioumax = iou
                            macth = obj

            if show_animation:
                status = "NO MATCH FOUND!"

            min_overlap = MINOVERLAP

            if specific_iou_flagged:
                if class_name in class_iou:
                    index = class_iou.index(class_name)
                    min_overlap = float(iou_list[index])

            if ioumax >= min_overlap:
                if not bool(macth["used"]):
                    # 匹配成功
                    tp[idx] = 1
                    match['used'] = True
                    result_dictionary[class_name] += 1

                    # 更新 .json 文件
                    with open(tmp_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
                    if show_animation:
                        status = "MATCH!"

                else:
                    # 匹配失败
                    fp[idx] = 1
                    if show_animation:
                        status = "REPEATED MATCH!"

            else:
                # 匹配失败
                fp[idx] = 1
                if ioumax > 0:
                    status = "INSUFFICIENT OVERLAP"

            # draw image to show animation
            # if show_animation:

        # 计算 precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val

        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val

        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / class_counter[class_name]

        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_ap += ap

        text = "{0:.2f}%".format(
            ap * 100) + " = " + class_name + " AP  "  # class_name + " AP = {0:.2f}%".format(ap*100)

        # 将结果写入 result文件夹中
        rounded_prec = ['%.2f' % elem for elem in prec]
        rounded_rec = ['%.2f' % elem for elem in rec]
        results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

        if not args.quiet:
            print(text)
        ap_dictionary[class_name] = ap

        # 画图
        if draw_plot:
            plt.plot(rec, prec, '-o')
            # add a new penultimate point to the list (mrec[-2], 0.0)
            # since the last line segment (and respective area) do not affect the AP value
            area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
            area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
            plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
            # set window title
            fig = plt.gcf()  # gcf - get current figure
            fig.canvas.set_window_title('AP ' + class_name)
            # set plot title
            plt.title('class: ' + text)
            # plt.suptitle('This is a somewhat long figure title', fontsize=16)
            # set axis titles
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # optional - set axes
            axes = plt.gca()  # gca - get current axes
            axes.set_xlim([0.0, 1.0])
            axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
            # Alternative option -> wait for button to be pressed
            # while not plt.waitforbuttonpress(): pass # wait for key display
            # Alternative option -> normal display
            # plt.show()
            # save the plot
            fig.savefig(results_files_path + "/classes/" + class_name + ".png")
            plt.cla()  # clear axes for next plot

    if show_animation:
        cv2.destroyAllWindows()

    results_file.write("\n# mAP of all classes\n")
    mAP = sum_ap / class_num
    text = "mAP = {0:.2f}%".format(mAP * 100)
    results_file.write(text + "\n")
    print(text)

# 删除 tmp文件夹
shutil.rmtree(tmp_files_path)

# 统计所有的预测值
# 每个类的预测值
pred_counter = {}
for txt_file in predicted_files_list:
    lines_list = file_to_list(txt_file)
    for line in lines_list:
        class_name = line.split()[0]
        if class_name in args.ignore:
            continue

        if class_name in pred_counter:
            pred_counter[class_name] += 1
        else:
            pred_counter[class_name] = 1

pred_classes = list(pred_counter.keys())

# 画图
if draw_plot:
    window_title = "Ground-Truth Info"
    plot_title = "Ground-Truth\n"
    plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(class_num) + " classes)"
    x_label = "Number of objects per class"
    output_path = results_files_path + "/Ground-Truth Info.png"
    to_show = False
    plot_color = 'forestgreen'
    draw_plot_func(
        class_counter,
        class_num,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        '',
    )


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions',
                 left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf()  # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)  # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


# 计算每个类的正确匹配数
for class_name in pred_classes:
    if class_name not in class_names:
        result_dictionary[class_name] = 0

"""
 Plot the total number of occurences of each class in the "predicted" folder
"""
if draw_plot:
    window_title = "Predicted Objects Info"
    # Plot title
    plot_title = "Predicted Objects\n"
    plot_title += "(" + str(len(predicted_files_list)) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = results_files_path + "/Predicted Objects Info.png"
    to_show = False
    plot_color = 'forestgreen'
    true_p_bar = result_dictionary
    draw_plot_func(
        pred_counter,
        len(pred_counter),
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        true_p_bar
    )

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if draw_plot:
    window_title = "mAP"
    plot_title = "mAP = {0:.2f}%".format(mAP * 100)
    x_label = "Average Precision"
    output_path = results_files_path + "/mAP.png"
    to_show = True
    plot_color = 'royalblue'
    draw_plot_func(
        ap_dictionary,
        class_num,
        window_title,
        plot_title,
        x_label,
        output_path,
        to_show,
        plot_color,
        ""
    )
