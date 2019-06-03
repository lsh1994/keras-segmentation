<font color=red>本项目不再更新。有个令人迷惑的地方，请参考issues5。谢谢支持。</font>
## 实验环境
win10；python3.6.7 ；
  
Item     | Value
-------- | -----
keras | 2.2.4  
tensorflow-gpu | 1.10.0

实验参考：  
https://github.com/divamgupta/image-segmentation-keras  
https://github.com/ykamikawa/SegNet  
  
实验数据：  
https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing  
实验数据或者：  
https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid

## 文件结构  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181218212010847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)

`python  visualizeDataset.py` 可视化样本：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181113165336706.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)

`python train.py`:训练  
`python predict.py`:预测  
请自行修改代码中相关参数切换模型，或者克隆历史版本。

## 详细描述

### FCN32

[文章参考](https://blog.csdn.net/nima1994/article/details/84031759)  
可视化结果：  
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018111410255134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)

### FCN8

[文章参考](https://blog.csdn.net/nima1994/article/details/84062253)  
可视化结果：  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181114103306961.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbWExOTk0,size_16,color_FFFFFF,t_70)


### SegNet
[文章参考](https://blog.csdn.net/nima1994/article/details/85079510)  

### U-Net
[文章参考](https://blog.csdn.net/nima1994/article/details/86300172)


## 注意
非论文的原始实现。  