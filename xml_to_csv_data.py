'''
@ moedl :xml_to_csv_data.py
@ brief : 该模块实现将xml数据转为csv文件
@ 参数：image_path 设置训练集(trian)的路径或测试集(test)的路径
@ 参数：save_csv   保存训练集或测试集的CSV路径，eg：
  image_path = os.path.join(os.getcwd(), 'image测试/train')  #
  save_csv=os.path.join(os.getcwd(), 'image','train.csv')
'''
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

#train数据的目录
print("project_dir=",os.getcwd())                             #当前工作目录
train_image_path = os.path.join(os.getcwd(), 'dataset/train')   #训练集的路径
train_save_csv=os.path.join(os.getcwd(), 'dataset','train.csv')#保存csv文件的路径

#test数据的目录
test_image_path = os.path.join(os.getcwd(), 'dataset/test')    #测试集的路径
test_save_csv=os.path.join(os.getcwd(), 'dataset','test.csv')  #保存csv文件的路径

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    #转换train图片数据
    train_xml_df = xml_to_csv(train_image_path)
    train_xml_df.to_csv(train_save_csv, index=None)
    print('Successfully converted xml to csv(train):',train_save_csv)

    #转换test图片数据
    test_xml_df = xml_to_csv(test_image_path)
    test_xml_df.to_csv(test_save_csv, index=None)
    print('Successfully converted xml to csv(test):',test_save_csv)
main()