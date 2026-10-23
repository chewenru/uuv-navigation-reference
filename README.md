# 水下机器人多候选导航参考实现

本项目提供一套模块化的水下机器人局部导航实现，便于理解、复用和接入已有控制系统。
核心流程涵盖运动候选生成、约束修正、视觉世界模型预测、风险评估和控制参考输出。

**局部运动意图 → 多候选采样 → 参考约束检查 → 视觉世界模型预测 → 风险选择 → 控制器参考输出**

## 模块

- `sampling.py`：围绕局部运动方向进行径向分层与三维角度采样。
- `constraints.py`：检查动作约束，并按显式设置修正速度上限。
- `reference.py`：生成共享的位置、姿态、速度及加速度参考。
- `geometry.py`：艇体包围盒与障碍体素的整段路径净空计算。
- `world_model.py`：视觉历史、传感器历史及候选参考条件的循环预测网络。
- `selection.py`：参考约束过滤、预测后果评估与候选排序。
- `controller.py`：将选中候选的同一参考发送给控制器。
- `pipeline.py`：串联候选、预测与选择。

## 安装与查看

使用 Python 3.9+，在本目录运行：

```bash
python -m pip install -e .
python -m examples.candidates
python -m unittest discover -s tests -v
```

`examples.candidates` 展示 32 个三维运动候选及一段共享参考，不连接机器人。
`examples/connect.py` 提供应用侧接线函数，接入观测、训练好的模型与控制器发送函数。

## 方法流程

1. **输入局部意图**：任务模块或视觉语言模型给出机体 FLU 坐标下的目标偏移。
2. **生成多候选**：在合法距离区间分层采样，沿方向圆锥构造相对的三维角度扰动。
3. **约束与修正**：检查位移、转角、速度和时长；`constrain(..., repair_speed=True)`
   显式启用速度上限修正，其余不合规候选直接报错。生成器默认只生成合法候选。
4. **共享运动参考**：五次多项式给出位置、速度、加速度；也保留原生参考网格读取函数。
   预测和控制使用同一份左端保持采样网格。
5. **预测执行后果**：CNN 编码 RGB 历史，GRU 融合传感器历史，再由候选参考条件驱动
   循环状态预测；输出传感器状态序列、位移序列，以及可选未来 RGB。
6. **几何与风险选择**：同时检查参考路径和预测执行路径；艇体由多个包围盒表示，
   障碍用占据体素表示，段间净空包含平移与转动扫掠修正。按风险、修正代价、原始排序、ID 排序。
7. **发送参考**：选中后按同一绝对起点播放，通过传入的 `publish(message)` 接入已有 PID 或控制器。

## 主要接口

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `generate_candidates` | 局部方向、候选数、随机种子、采样范围和动作限制 | `Candidate` 列表 |
| `constrain` | 候选与动作限制 | 合法候选及累计修正代价 |
| `build_reference` | 候选、从零开始的秒时间网格 | `Reference` |
| `WorldModel.forward` | `History`、共享参考列表 | 按候选 ID 绑定的预测序列 |
| `select_candidate` | 候选、参考路径、预测路径、艇体与障碍体素 | 候选 ID 和逐候选净空 |
| `decide` | 当前观测、模型、候选、几何和传感器安装姿态 | 选择结果及同一份参考对象 |
| `ReferencePlayer.tick` | 当前纳秒时间戳 | 发给控制器的参考消息 |

### 坐标与输入

几何、候选参考和预测轨迹统一在**决策起点的固定 FLU 坐标系**中：前、左、上。
`ReferencePlayer` 用该起点的实际位置和姿态变换到控制器指定坐标系。
传感器安装旋转使用 XYZW 四元数，显式将 IMU/DVL 原生轴转换到机体 FLU。

传感器历史 `History.values` 为 `[1,T,26]`：

- `0:3`：DVL 速度，m/s；`3:6`：IMU 角速度，rad/s。
- `6:9`：IMU 线加速度，m/s²；`9`：压力，kPa；`10`：高度，m。
- `11:19`：之前的 8 路执行器值。
- `19:23`：固定起点坐标中的姿态四元数 XYZW；`23:26`：速度 XYZ。

RGB 为 `[1,T,3,64,96]` 的 uint8。历史保留有效掩码、源时间、可用时间和采样时刻；
网络只接收当时已经可用的观测。`state_mean` 与 `state_scale` 使用训练时的 19 通道统计。

每个参考区间编码 29 维条件：位置3、姿态4、速度6、加速度6、控制增量6、
初始误差3、经过时间1。初始误差维度保持掩码，候选的未来反馈增量为零。

几何字典：

```python
environment = {
    "decision_ns": observation_time_ns,
    "frame_id": "anchor_flu",
    "voxel_centres_m": measured_voxel_centres,
    "voxel_size_m": voxel_size_m,
}
vehicle = {
    "primitives": [
        {"shape": "box", "size_xyz_m": body_size,
         "origin_xyz_m": body_offset, "origin_rpy_rad": body_rotation}
    ]
}
```

`measured_voxel_centres` 需要由调用方完成测量时刻到当前固定起点的配准。
`margin_m` 由应用设置。没有合适候选时，返回的候选 ID 和参考为 `None`，由调用方处理。

### 模型接入

```python
from uuv_navigation.world_model import WorldModel

model = WorldModel.from_checkpoint("your_model.pt", device="cuda")
```

权重文件包含 `config` 和 `state_dict`：`config` 对应 `WorldModel` 构造参数，
包括训练使用的 `state_mean`、`state_scale`、隐藏维数及输出尺度；`state_dict` 对应
本模块的网络层。调用者提供训练好的模型参数，不自动下载、训练或初始化替代预测。

位移输出支持直接累积位移头，或通过 `velocity_scale_m_s` 配置速度积分头；
`reference_velocity_prior=True` 在后一种配置中加入参考速度先验。
风险路径由预测的陀螺仪与 DVL 状态按原积分方式读出，与位移输出头独立。

### 控制器接入

`examples.connect.select_reference_player` 返回 `selection, player`。
将观测起点的实际位姿和时间传入；控制循环调用 `player.tick(now_ns)`。
发送内容包含候选 ID、起点时间、当前时间、坐标系、位置、姿态、六维速度与加速度。
`publish` 是应用现有发送函数，可以接 ROS 消息或其他控制接口；库本身不启动服务。

首个 `tick` 必须使用与预测起点相同的时刻，后续按绝对时间播放；起点不一致直接报错，
不会将已过去的预测参考重新从零发送。
