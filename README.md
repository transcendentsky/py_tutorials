**Py-Tutorials (example)**
======== 
<!-- Description -->
![Trans](.github/corgi1.jpg)

Paper link: arxiv.org/xxx

Abstract: There are some demo codes and test codes
一些python各个功能的示例, 以及测试代码

# Usage

## Environment
python == 3.5/3.6, 
pytorch >= 1.1.0, 
torchvison >= 0.6


```
pip install -r requirements.txt
```

## Data preparation
We train/test our model on Datasets (e.g. [KiTS19](http://xxx.org/xxx/xx) ) 
<!-- or You can download from link: xxx -->

We expect the directory structure to be the following:
```
path/to/kits19
    data/
        case_00xxx/
            imaging.nii.gz
            segmentation.nii.gz
        case_00xxx/
        ...
```
## Training
To train our model, run this scripts
```
python -m scripts.train --epochs 300 --data_path path/to/kits19 
```
To evaluate our model, run this scripts
```
python -m scripts.test --data_path path/to/kits19 --resume xxx-model.pth
```

# Citation
Please cite our paper if it helps you.
```
@proceeding{
    title
}
```
# License
This code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Contribution
We actively welcome your pull requests! feel free!


<!-- # 问医生的一些问题
# 背光对于看x片是否有很大影响
# x片上的顺序/序号的选择方式
# 在什么情况下需要重新拍摄，动态模糊，遮挡/大角度弯曲
# 阅片的重点在哪里，较细节的斑点，或者某部位的形变 （低分辨率的影响）
# 每个部位的x片诊断方式有何不同之处
# 16个照片有没有重要性等级区别， -->