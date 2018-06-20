# encoding: utf-8
'''
@author: By Alian
@desc:   精简了物体检测模型的测试方法代码
'''
import numpy as np
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

## This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util


#download model :http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT='detection_model/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb'#模型路径
# PATH_TO_CKPT='detection_model/my_model/pb/frozen_inference_graph.pb'#模型路径
# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS='data/pascal_label_map.pbtxt'#label路径

#PATH_TO_CKPT='detection_model/car_model/frozen_inference_graph.pb'#模型路径
#PATH_TO_LABELS='detection_model/car_model/label_map.pbtxt'#label路径
#image_path='test_images/dog29.jpg'#测试图片

PATH_TO_CKPT='models/pb/frozen_inference_graph.pb'#模型路径
PATH_TO_LABELS='data/label_map.pbtxt'#label路径
image_path='test_images/6.jpg'#测试图片


NUM_CLASSES = 2
#Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)
      plt.show()