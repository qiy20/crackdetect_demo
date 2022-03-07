# VERSION2 

## 更新内容

1. 移除models权重，下载使用清华云盘（https://cloud.tsinghua.edu.cn/d/fac2b47c566b4c2ca68d/）

2. 更新model_cfg文件，记录模型版本参数

   |                                              | 参数                                                         |
   | -------------------------------------------- | ------------------------------------------------------------ |
   | 训练x方式                                    | 联合训练，$\lambda = ?$                                      |
   | base_model                                   | yolov5x_v6.0.0                                               |
   | grid_neck                                    | PANnet                                                       |
   | 训练超参                                     | lr0: 0.001<br/>lrf: 0.1<br/>……..<br/>mixup: 0.0<br/>copy_paste: 0.0 |
   | box map                                      | TransverseCrack: 0.336<br/>LongitudinalCrack: 0.274<br/>AlligatoCrack: 0.423<br/>StripRepair: 0.785<br/>Marking: 0.958<br/>Joint: 0.081 |
   | grid map                                     | Crack: 0.760<br/>Repair: 0.943<br/>Marking: 0.972            |
   | Flops                                        | 220.1 GFLOPs                                                 |
   | inference time（bs=32 for single TiTAN RTX） | 1.87s                                                        |
   | dataset                                      | train: 13896<br/>val:3475<br/>test: 5014                     |

3. 不再对宽高比>1.5的图进行拆分，将整张照片作为输入

4. 更新权重加载方式，兼容两个版本的模型

## TODO

- [ ] 固定数据集，统一测试模型指标
- [ ] 解决grid中crack具体属性依赖box获取



# VERSION1s

## 1.Requirements

### CPU配置

直接安装以下包即可：

```bash
numpy==1.20.3
onnxruntime==1.9.0
opencv-python==4.5.3.56
torch==1.9.1
torchvision==0.10.1
```

### GPU配置

首先需要配置CUDA和CUDNN，要求：

```bash
CUDA==11.1
CUDNN>=8.0.2
```

可以使用原生配置，也可以使用conda进行安装，以使用conda为例，新建虚拟环境之后：

```bash
conda install -c anaconda cudnn
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install onnxruntime-gpu
```

运行上述三条命令即可。

## 2.deep learning模型（models/s.onnx,models/x.onnx）

![model.png](https://github.com/qiy20/crackdetect_demo/blob/main/model.png?raw=true)

### 输入：

1. 张量[batch_size,chanels,img_h,img_w]，其中chanels=3为固定的，（batch_size,img_h,img_w）为dynamic axis。
2. 建议取值：batch_size根据GPU显存确定，img_h 和img_w建议取640。

### 输出：

1. 输出由两部分组成，box_pred和grid_pred。
2. box_pred的维度为 [batch_size, nc, 11]，其中nc为anchors的数量与输出图像尺寸有关，11为**[ 4（box reg）+1（objectness）+6（classify）]**
3. grid_pred的维度为[batch_size ,gx, gy, 3]

## 3.CrackDetector类

包含四部分内容，调用接口run()：

1. preprocess：读取图片，解码图片，将宽图（大车的图）分为左右两张，resize
2. model infer process：onnx模型推断
3. postprocess: NMS，RAS，将grid输出转换为box格式，match两种结果，将宽图的结果拼接
4. export：导出结果，包含image和txt

## 4.运行时间

以x模型, batch_size=32为例：

![time_consuming.png](https://github.com/qiy20/crackdetect_demo/blob/main/time_consuming.png?raw=true)

除推断过程外，耗时比较严重的是read和decode，这里尝试了多进程读取和解码图片，性能提升很小。

实际部署时可能需要提前将后几个batch数据缓存到内存中。

## 5.效果展示

RAS之前与之后比较：

![before&after_RAS.png](https://github.com/qiy20/crackdetect_demo/blob/main/before&after_RAS.png?raw=true)

最终效果展示：

![res.png](https://github.com/qiy20/crackdetect_demo/blob/main/res.png?raw=true)

