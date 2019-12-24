<h1 align="center"><a href="https://github.com/lsh1994/keras-segmentation" target="_blank">keras-segmentation</a></h1>

<p align="center">
Implementation is not original papers. The purpose of this project is to get started with semantic segmentation and master the basic process.
</p>

<font color=red>FCN32/8、SegNet、U-Net [Model published](https://github.com/lsh1994/keras-segmentation/releases)。There is a confusing place, please refer to [issues 5](https://github.com/lsh1994/keras-segmentation/issues/5) and so on.Thank you for your support.</font> 

[中文说明](readme_zh.md)

## Environment
  
Item     | Value | Item     | Value
:---: | :---: | :---: | :---: 
keras | 2.2.4  | OS | win10
tensorflow-gpu | 1.10/1.12 | Python| 3.6.7

## Reference
  
https://github.com/divamgupta/image-segmentation-keras  
https://github.com/ykamikawa/SegNet  
  
Data：  
https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing  
Data or：  
https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

## Project Strcutre  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181218212010847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)

`python  visualizeDataset.py`: Visual samples  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181113165336706.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)

`python train.py`: Execution train    
`python predict.py`: Execution predict  

You can modify the parameter in project switching model or cloning the historical version.

## About

### FCN32

Visualization results:    
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018111410255134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)

### FCN8

Visualization results:   
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181114103306961.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)


### SegNet

### U-Net


  