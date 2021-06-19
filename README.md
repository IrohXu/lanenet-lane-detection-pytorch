# Lanenet-Lane-Detection (pytorch version)
  
Use pytorch to implement a Deep Neural Network for real time lane detection mainly based on the IEEE IV conference paper "Towards End-to-End Lane Detection: an Instance Segmentation Approach".You can refer to their paper for details https://arxiv.org/abs/1802.05591. This model consists of ENet encoder, ENet decoder for binary semantic segmentation and ENet decoder for instance semantic segmentation using discriminative loss function.  

The main network architecture is:  

## Generate training set/validation set/test tet  
First, download tusimple dataset here[https://github.com/TuSimple/tusimple-benchmark/issues/3].  
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
--dataset
----clips
----label_data_0313.json
----label_data_0531.json
----label_data_0601.json
----test_label.json
```

Future work:  
