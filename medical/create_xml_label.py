import xml.etree.ElementTree as ET

def createXMLlabel(savedir,objectnum, bbox, classname, foldername='0',filename='0', path='0', database='road', width='400', height='600',depth='3', segmented='0', pose="Unspecified", truncated='0', difficult='0'):
    # 创建根节点
    root = ET.Element("annotation")

    # 创建子节点
    folder_node = ET.Element("folder")
    folder_node.text = foldername
    # 将子节点数据添加到根节点
    root.append(folder_node)

    file_node = ET.Element("filename")
    file_node.text = filename
    root.append(file_node)
    path_node = ET.Element("path")
    path_node.text = path
    root.append(path_node)

    source_node = ET.Element("source")
    # 也可以使用SubElement直接添加子节点
    db_node = ET.SubElement(source_node, "database")
    db_node.text = database
    root.append(source_node)

    size_node = ET.Element("size")
    width_node = ET.SubElement(size_node, "width")
    height_node = ET.SubElement(size_node, "height")
    depth_node = ET.SubElement(size_node, "depth")
    width_node.text = width
    height_node.text = height
    depth_node.text = depth
    root.append(size_node)

    seg_node = ET.Element("segmented")
    seg_node.text = segmented
    root.append(seg_node)

    for i in range(objectnum):
        newEle = ET.Element("object")
        name = ET.Element("name")
        name.text = classname
        newEle.append(name)
        pose_node = ET.Element("pose")
        pose_node.text = pose
        newEle.append(pose_node)
        trunc = ET.Element("truncated")
        trunc.text = truncated
        newEle.append(trunc)
        dif = ET.Element("difficult")
        dif.text = difficult
        newEle.append(dif)
        boundingbox = ET.Element("bndbox")
        xmin = ET.SubElement(boundingbox, "xmin")
        ymin = ET.SubElement(boundingbox, "ymin")
        xmax = ET.SubElement(boundingbox, "xmax")
        ymax = ET.SubElement(boundingbox, "ymax")
        xmin.text = str(bbox[i][1])
        ymin.text = str(bbox[i][0])
        xmax.text = str(bbox[i][3])
        ymax.text = str(bbox[i][2])
        newEle.append(boundingbox)
        root.append(newEle)

    ImageID = filename.split('.')[0]
    # 创建elementtree对象，写入文件
    tree = ET.ElementTree(root)
    tree.write(savedir + '/'+ ImageID + ".xml")


imagedir = r('D:\test_dataset\labelimage')
saveXMLdir = r('D:\test_dataset\Annotations')

if os.path.exists(saveXMLdir) is False:
    os.mkdir(saveXMLdir)

for root, _, fnames in sorted(os.walk(imagedir)):
    for fname in sorted(fnames):
        labelpath = os.path.join(root, fname)
        labelimage = misc.imread(labelpath)
        # 得到label图上的boundingingbox和数量
        objectnum, bbox = getboundingbox(labelimage)
        # label图 命名格式为 ImgeID_classname.png
        labelfilename = labelpath.split('\\')[-1]
        ImageID = labelfilename.split('.')[0].split('_')[0]
        classname = labelfilename.split('.')[0].split('_')[1]
        origin_image_name = ImageID +'.jpg'
    
        # 一些图片信息
        foldername = 'test_dataset'
        path  ='\\'.join(imagedir.split('\\')[:-1]) + '\\JPEGImage\\'+ origin_image_name
        database = 'Unknown'
        width = str(labelimage.shape[0])
        height = str(labelimage.shape[1])
        depth = str(labelimage.shape[2])
        
        createXMLlabel(saveXMLdir,objectnum, bbox, classname, foldername=foldername,filename=origin_image_name, path=path,
                       database=database, width=width, height=height,depth=depth, segmented='0', pose="Unspecified",
                       truncated='0', difficult='0')