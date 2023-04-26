# Image_Recognition_WebGUI
基于ADNI数据集的阿尔兹海默智能诊断Web应用：AI人工智能图像识别-Pytorch；可视化Web图形界面-Pywebio； nii医学影像识别。100%纯Python 

## 功能简介
* 1, 根据脑部MRI医学影像智能诊断阿尔兹海默病
* 2, 绘制参数相关性热力图
* 3, 使用纯python编写，轻量化，易复现，易部署

## 界面展示
* 进入web界面
![image](.\readme_static\readme_img\4.png)
* 点击"使用demo.nii"，可以使用默认的demo图像测试识别功能
![image](.\readme_static\readme_img\3.png)
* 也可以自己上传医学影像
![image](.\readme_static\readme_img\9.png)
* 点击"查看图像"，渲染参数热力图
![image](.\readme_static\readme_img\5.png)
![image](.\readme_static\readme_img\6.png)
* 根据上传的图像，生成参数相关性热力图
![image](.\readme_static\readme_img\7.png)

## 如何使用
python版本3.9

先安装依赖
> pip install -r requirement.txt

demo01.py是项目入口，运行此文件即可启动服务器
> python demo01.py

复制链接到浏览器打开
![](.\readme_static\readme_img\10.png) 
点击”Demo“即可进入Web界面
![](.\readme_static\readme_img\11.png)


## 项目结构
```
└─Image_Recognition_WebGUI
    ├─data
    │  └─model_save
    ├─imgs
    │  ├─img_hot
    │  ├─img_merge
    │  └─img_raw
    ├─nii
    ├─readme_static
    │  └─readme_img
    └─run_logs
```
* data文件夹存放部分静态资源，其中的model_save文件夹存放训练好的模型
* imgs文件夹存放渲染的图片
* nii文件夹存放用户上传的医学影像数据
* readme_static存放readme文档中用的静态资源
* run_logs存放用户访问日志

ref:  https://github.com/moboehle/Pytorch-LRP

数据集:https://adni.loni.usc.edu"