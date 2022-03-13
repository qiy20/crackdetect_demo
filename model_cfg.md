# Model_Cfg

## x_version2.onnx

|                                              | 参数                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| 训练x方式                                    | 联合训练，$\lambda = ?$                                      |
| base_model                                   | yolov5x_v6.0.0                                               |
| grid_neck                                    | PANnet                                                       |
| 训练超参                                     | lr0: 0.001<br/>lrf: 0.1<br/>momentum: 0.937<br/>weight_decay: 0.0005<br/>warmup_epochs: 3.0<br/>warmup_momentum: 0.8<br/>warmup_bias_lr: 0.1<br/>grid: 0.01<br/>grid_pw: 1.0<br/>box: 0.05<br/>cls: 0.05<br/>cls_pw: 1.0<br/>obj: 0.1<br/>obj_pw: 1.0<br/>iou_t: 0.2<br/>anchor_t: 4.0<br/>fl_gamma: 0.0<br/>fl_gamma_grid: 0.0<br/>hsv_h: 0.015<br/>hsv_s: 0.7<br/>hsv_v: 0.4<br/>degrees: 0.0<br/>translate: 0.0<br/>scale: 0.0<br/>shear: 0.0<br/>perspective: 0.0<br/>flipud: 1.0<br/>fliplr: 1.0<br/>mosaic: 1.0<br/>mixup: 0.0<br/>copy_paste: 0.0 |
| box map                                      | TransverseCrack: 0.336<br/>LongitudinalCrack: 0.274<br/>AlligatoCrack: 0.423<br/>StripRepair: 0.785<br/>Marking: 0.958<br/>Joint: 0.081 |
| grid map                                     | Crack: 0.760<br/>Repair: 0.943<br/>Marking: 0.972            |
| Flops                                        | 220.1 GFLOPs                                                 |
| inference time（bs=32 for single TiTAN RTX） | 1.87s                                                        |
| dataset                                      | train: 13896<br/>val:3475<br/>test: 5014                     |
| threshold_interval                           | grid: [0.7, 0.9]<br/>box:[0.1, 0.3]                          |

## s_version1.onnx

|                                              | 参数                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| 训练x方式                                    | 先训练box，冻结box相关，再训练grid                           |
| base_model                                   | yolov5s_v5.0.0                                               |
| grid_neck                                    | PANnet                                                       |
| 训练超参                                     | lr0: 0.01<br/>lrf: 0.2<br/>momentum: 0.937<br/>weight_decay: 0.0005<br/>warmup_epochs: 3.0<br/>warmup_momentum: 0.8<br/>warmup_bias_lr: 0.1<br/>box: 0.05<br/>cls: 0.5<br/>cls_pw: 1.0<br/>obj: 1.0<br/>obj_pw: 1.0<br/>iou_t: 0.2<br/>anchor_t: 4.0<br/>fl_gamma: 0.0<br/>hsv_h: 0.015<br/>hsv_s: 0.7<br/>hsv_v: 0.4<br/>degrees: 0.0<br/>translate: 0.1<br/>scale: 0.5<br/>shear: 0.0<br/>perspective: 0.0<br/>flipud: 0.0<br/>fliplr: 0.5<br/>mosaic: 1.0<br/>mixup: 0.0 |
| box map                                      |                                                              |
| grid map                                     |                                                              |
| Flops                                        |                                                              |
| inference time（bs=32 for single TiTAN RTX） |                                                              |
| dataset                                      |                                                              |

## x_version1.onnx

|                                              | 参数                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| 训练x方式                                    | 先训练box，冻结box相关，再训练gridbase                       |
| base_model                                   | yolov5x_v5.0.0                                               |
| grid_neck                                    | PANnet                                                       |
| 训练超参                                     | lr0: 0.01<br/>lrf: 0.2<br/>momentum: 0.937<br/>weight_decay: 0.0005<br/>warmup_epochs: 3.0<br/>warmup_momentum: 0.8<br/>warmup_bias_lr: 0.1<br/>box: 0.05<br/>cls: 0.5<br/>cls_pw: 1.0<br/>obj: 1.0<br/>obj_pw: 1.0<br/>iou_t: 0.2<br/>anchor_t: 4.0<br/>fl_gamma: 0.0<br/>hsv_h: 0.015<br/>hsv_s: 0.7<br/>hsv_v: 0.4<br/>degrees: 0.0<br/>translate: 0.1<br/>scale: 0.5<br/>shear: 0.0<br/>perspective: 0.0<br/>flipud: 0.0<br/>fliplr: 0.5<br/>mosaic: 1.0<br/>mixup: 0.0 |
| box map                                      |                                                              |
| grid map                                     |                                                              |
| Flops                                        |                                                              |
| inference time（bs=32 for single TiTAN RTX） |                                                              |
| dataset                                      |                                                              |



