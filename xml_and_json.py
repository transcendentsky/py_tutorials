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

import json
def get_json(filename):
    with open(filename, 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
        print(len(load_dict))
    # load_dict['smallberg'] = [8200, {1: [['Python', 81], ['shirt', 300]]}]
    # print(load_dict)

def findfile():
    files = os.listdir('.')
    for file in files:
        if file[-5:] == '.json':
            print(file)
            get_json(file)

if __name__ == '__main__':
    findfile()