"""
Usage:
  # From tensorflow/models/
  # Create train data:
python cvs_tf_record.py --csv_input=image/train.csv  --output_path=data/train.record
  # Create test data:
python cvs_tf_record.py --csv_input=image/test.csv  --output_path=data/test.record
  需要修改三处
  os.chdir('D:\\python3\\models-master\\research\\object_detection\\')
  path = os.path.join(os.getcwd(), 'images/train')
  def class_text_to_int(row_label): #对应的标签返回一个整数，后面会有文件用到
    if row_label == 'ZJL':
        return 1
    elif row_label == 'CYX':
        return 2
    else:
        None
"""

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# os.chdir('D:\\Anaconda3\\envs\\models\\research\\object_detection\\')

#训练
# image_dir = os.path.join(os.getcwd(), 'image\\train')          # 训练图片所在目录
# csv_path = os.path.join(os.getcwd(), 'image\\train.csv')      # 训练图像的csv文件的路径
# record_path=os.path.join(os.getcwd(), 'data\\train.record')   # 输出：保存训练record文件的位置
#测试
image_dir = os.path.join(os.getcwd(), 'dataset\\test')          # 测试图片所在目录
csv_path = os.path.join(os.getcwd(), 'dataset\\test.csv')      # 测试图像的csv文件的路径
record_path=os.path.join(os.getcwd(), 'data\\test.record')   # 输出：保存测试record文件的位置


#通过命令行输入路径参数
flags = tf.app.flags
flags.DEFINE_string('image_dir',image_dir, 'Path to the image dir')
flags.DEFINE_string('csv_path',csv_path, 'Path to the CSV input')
flags.DEFINE_string('record_path', record_path, 'Path to output record_path')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'dog':
        return 1
    elif row_label == 'person':
        return 2
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    per_path=os.path.join(path, '{}'.format(group.filename))
    if os.path.exists(per_path) is False:
        print("image is not exist",per_path)
        return False,None
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    image_dir = FLAGS.image_dir
    csv_path = FLAGS.csv_path
    record_path =FLAGS.record_path

    writer = tf.python_io.TFRecordWriter(FLAGS.record_path)
    examples = pd.read_csv(FLAGS.csv_path)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(FLAGS.record_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()