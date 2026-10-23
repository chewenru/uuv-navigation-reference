# 实现来源说明

- 候选采样、五次多项式参考轨迹、参考网格播放、艇体包围盒扫掠净空计算及候选排序，
  提取或适配自已有项目实现。
- 视觉与循环预测核心保留卷积神经网络（CNN）、历史特征融合、条件门控循环单元（GRU）、
  状态增量读出和位移预测头的计算方式；对外接口使用普通张量和参考轨迹对象。
- 随姿态变化的轴对齐包围盒（AABB）范围使用 `abs(R) * half_size` 计算，
  对应 Bullet 的 `btTransformAabb`（`LinearMath/btAabbUtil2.h`），Stonefish 也采用该实现。
- 数值与张量计算使用 NumPy、SciPy 和 PyTorch。本项目不附带 ROS 或仿真器源码。
