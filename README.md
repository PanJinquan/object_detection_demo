# object_detection_demo说明
## 一：目录结构说明
--object_detection_demo
|_____data   ：存放record文件、配置文件和标签文件</br>
|_____dataset：存放测试和训练的图片数据集</br>
|_____models ：存放训练过程中模型文件</br>
|_____test_images ：存放用于模型测试的图片</br>
|_____tools  ：数据集制作工具：请参考：https://github.com/PanJinquan/labelImg</br>

## 二：配置说明
首先把TensoFlow的models/research等相关模块导入的Python搜索路径中，避免查不到文件出错的问题，方法是
在python的Lib\site-packages目录下，新建一个*.pth文件（文件名随意），然后把models/research、models/object_detection和models/research/slim的绝对路径编辑到文件中：
> D:\Anaconda3\envs\models\research
> D:\Anaconda3\envs\models\research\object_detection
> D:\Anaconda3\envs\models\research\slim

PS：相关的配置文件，凡是用到相对路径，都是工作在当前object_detection_demo目录下

## 三：训练流程步骤：
**1.准备训练图集（train dataset）和测试图集（test dataset）**

**2.使用“labelImg”制作训练train数据和测试test数据，保存为.xml文件 **

**3.将xml数据转为csv文件，方便转换数据**

**4.将数据转为record文件：**
>  可以在“cvs_tf_record.py”文件中修改训练/测试文件路径，生成record文件
>  也可以使用命令行，进行转换：如</br>
  python cvs_tf_record.py    --image_dir=    --csv_path=   --record_path=

**5.新建标签映射文件data/label_map.pbtxt，其id从1开始：**

``` 
	item {
	 name: "dog"
	 id: 1
	 display_name: "dog"
	}
	item {
	 name: "person"
	 id: 2
	 display_name: "person"
	}
```
**6.把配置文件放在training文件夹下：training/ssd_mobilenet_v1_coco.config，如下修改：**

(1).修改训练数据的路径和数据标签路径
```
	train_input_reader: {
	  tf_record_input_reader {
		input_path: "data/train.record"
	  }
	  label_map_path: "data/label_map.pbtxt"
	}
```
(2).修改测试数据的路径和数据标签路径
```
	eval_input_reader: {
	  tf_record_input_reader {
		input_path: "data/test.record"
	  }
	  label_map_path: "data/label_map.pbtxt"
	  shuffle: false
	  num_readers: 1
	}
```
(3).修改num_classes标签类别数，这里只两种类别，所以num_classes: 2
```
	  ssd {
		num_classes: 2
		box_coder {
		  faster_rcnn_box_coder {
			y_scale: 10.0
			x_scale: 10.0
			height_scale: 5.0
			width_scale: 5.0
		  }
		}
```
(4).（可选修改）batch_size是每次迭代的数据数，我这里设为1，当然也可以是8、16等任意数据，看你内存大小吧
```
	train_config: {
	  batch_size: 24
	  optimizer {
		rms_prop_optimizer: {
		  learning_rate: {
			exponential_decay_learning_rate {
			  initial_learning_rate: 0.004
			  decay_steps: 800720
			  decay_factor: 0.95
			}
		  }
		  momentum_optimizer_value: 0.9
		  decay: 0.9
		  epsilon: 1.0
		}
	  }
```
(5).（可选修改）变量fine_tune_checkpoint即微调检查点文件,用于指示以前模型的路径以获得学习，在应用转移学习上被使用。转移学习是一种机器学习方法，它专注于将从一个问题中获得的知识应用到另一个问题上。这里可以注释掉：
```
#fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
#from_detection_checkpoint: true
```
**7.开始训练模型**

  由于train.py在models/research/object_detection目录中，而我们的工程目录是models/research/object_detection_demo，因此需要注意路径问题
  在object_detection_demo工程目录时，可直接使用以下命令，进行训练：注意train.py的路径：
>  python ../object_detection/train.py --logtostderr --train_dir=models/  --pipeline_config_path=data/ssd_mobilenet_v1_coco.config </br>

  或者直接用train.py的绝对路径进行训练：
  
>  python D:/Anaconda3/envs/models/research/object_detection/train.py --logtostderr --train_dir=models/ --pipeline_config_path=data/ssd_mobilenet_v1_coco.config

**8.输出模型**

> python export_inference_graph.py \
> --input_type image_tensor \
> --pipeline_config_path data/ssd_mobilenet_v1_coco.config \
> --trained_checkpoint_prefix models/model.ckpt-110 \
> --output_directory models/pb

或者

> python ../object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path data/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix models/model.ckpt-110 --output_directory models/pb

**9.测试模型**

把模型测试的图片放在test_images文件夹中，运行object_detection_test.py或者object_detection_test_02.py文件


## 四：参考资料

[1] https://zhuanlan.zhihu.com/p/35854575
