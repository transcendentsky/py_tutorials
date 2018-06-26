#coding:utf-8
# 读取XML文件
import os
import xml.etree.ElementTree as ET
# from xml.etree.ElementTree import parse, Element

def get_xml_info(path):
    filenames = os.listdir(path)
    fnames = []
    all_boxes = []
    for filename in filenames:
#        if filename[-5] == '6' and filename[-3:] == 'xml':
        if filename[-3:] == 'xml':
            tree = ET.parse(os.path.join(path,filename))
            root = tree.getroot()
            fname = root.find('filename').text
#            print("fname: ", fname)
            fnames.append(fname)
            boxes = []
            for ob in root.iter('object'):
                for bndbox in ob.iter('bndbox'):
                    box = []
                    for l in bndbox:
                        box.append(int(l.text))
                boxes.append(box)
        all_boxes.append(boxes)
    return fnames, all_boxes

if __name__ == '__main__':
    print('???')

#####################################################################
import json
def get_json(filename):
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
        print(len(load_dict))
        print(load_dict[3][2])
    # load_dict['smallberg'] = [8200, {1: [['Python', 81], ['shirt', 300]]}]
    # print(load_dict)

def findfile():
    files = os.listdir('.')
    for file in files:
        if file[-5:] == '.json':
            print(file)
            get_json(file)

#####################################################################
import csv
with open('run_nomix_cifar100_mute_with_xavier_logs-tag-Test_1001_val_acc.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    # print(headers)
    for row in f_csv:
        print(row)

# if __name__ == '__main__':
#     findfile()