# Lanenet-Lane-Detection (基于pytorch的版本)

[English Version](https://github.com/IrohXu/lanenet-lane-detection-pytorch/blob/main/README.md)    

在本项目中，使用pyotrch复现了 IEEE IV conference 的论文 "Towards End-to-End Lane Detection: an Instance Segmentation Approach"，并对这篇论文的思想进行讨论。   
开发这一项目的初衷是，在github上开源的LaneNet项目数目较少，其中只有基于tensorflow 1.x的项目https://github.com/MaybeShewill-CV/lanenet-lane-detection 能够完整的实现作者论文中的思想，但是随着tensorflow 2.x的出现，基于tensorflow 1.x的项目在未来的维护将会越来越困难，很多刚入门深度学习同学也不熟悉tensorflow 1.x的相关功能。与此同时，pytorch的几个LaneNet项目或多或少都存在一些问题，且相关作者已经不再维护。   

LaneNet的网络框架:    
![NetWork_Architecture](./data/source_image/network_architecture.png)

## 生成用于训练和测试的Tusimple车道线数据集      
First, download tusimple dataset [here](https://github.com/TuSimple/tusimple-benchmark/issues/3).  
Then, run the following command to generate the training/ val/ test samples and the train.txt/ val.txt/ test.txt file.  
Generate training/ val/ test set:  
```
python tusimple_transform.py --src_dir path/to/your/unzipped/file --val True
```
Generate training/ test set:  
```
python tusimple_transform.py --src_dir path/to/your/unzipped/file --val False
```
path/to/your/unzipped/file should like this:  
```
|--dataset
|----clips
|----label_data_0313.json
|----label_data_0531.json
|----label_data_0601.json
|----test_label.json
```

Future work:  
