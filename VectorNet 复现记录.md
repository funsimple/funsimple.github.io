# VectorNet 复现记录

##  1. 数据处理

VectorNet的输入分为两部分，分别为目标历史轨迹与属性构成的vector和地图信息构成的vector。本次复现使用的数据集为[Argoverse daraset](https://www.argoverse.org/)，数据集的API地址为：[Argoverse API](https://github.com/argoai/argoverse-api)

- 目标历史轨迹

  一般而言，使用argoverse数据集，是用前2秒（20个轨迹点）的轨迹预测后3秒（30个轨迹点）的轨迹。需要预测的目标，轨迹可以转化成19 * N的向量，19是因为20个点可以构成19个向量，N是代表fearture个数，一般可以选择64或者128等，实际不够则用0补充。本次复现，选择128，各项feature为：

  ```python
  [pre_x, pre_y, x, y, timestamp, is_AV, is_AGENT, is_OTHERS, obj_id, traj_vector_id, 0, 0, ..., 0]
  traj_vector_id后补零至设置的feature num
  ```

  对于非预测的目标，轨迹点可能不足20个，比如可能为13 * N，需要补零向量至19 * N。

  最终目标历史轨迹处理后的数据的shape为： [obj_num, 19, feature_num]

- 地图信息

  argoverse的地图，每一个lane由10个点表示，所以可以转换成 9 * N 的向量，为了与轨迹对齐，也补零向量至19 * N。每个向量可以表示为：

  ```python
  [0, ... , pre_pre_x, pre_pre_y, is_intersection, turn_direction, has_traffic_control, obj_id, point_id, is_map, x, y, pre_x, pre_y]
  ```

最后，将两种vector拼起来，因此一个场景（单个excel）的数据可以处理成shape为[obj_num, 19, feature_num]的Tensor，obj_num在此处表示该场景下所有目标和lane的总和。

需要注意的是，不同场景下的obj_num不一样，这需要在训练的时候注意，具体处理方法可以参考[Dataloader](#Datalodar)使用。

























































## 技术细节

1. <span id = "Datalodar">关于pytorch的Dataloader</span>

   在处理

2. 