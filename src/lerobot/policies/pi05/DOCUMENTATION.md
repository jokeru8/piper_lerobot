# π₀.₅ (PI05) 完整文档

## 目录

1. [文件结构](#文件结构)
2. [文件功能概述](#文件功能概述)
3. [代码块功能详解](#代码块功能详解)
4. [关键代码块详细说明](#关键代码块详细说明)
5. [变量含义](#变量含义)
6. [调用顺序](#调用顺序)
7. [数据流](#数据流)
8. [架构图](#架构图)
9. [组件交互图](#组件交互图)
10. [关键概念说明](#关键概念说明)
11. [文件依赖关系](#文件依赖关系)

---

## 文件结构

```
pi05/
├── __init__.py              # 包初始化，导出主要类和函数
├── configuration_pi05.py     # 配置类定义
├── modeling_pi05.py         # 核心模型实现（1233行）
├── processor_pi05.py        # 数据预处理和后处理管道
└── README.md                # 模型说明文档
```

---

## 文件功能概述

### 1. `__init__.py` - 包初始化文件

**功能**：导出模块的主要类和函数，供外部使用

**导出内容**：
- `PI05Config`：配置类
- `PI05Policy`：策略类
- `make_pi05_pre_post_processors`：数据处理器创建函数

### 2. `configuration_pi05.py` - 配置类

**功能**：定义 PI05 模型的所有配置参数

**主要配置类别**：
- 模型架构参数
- Flow Matching 参数
- 训练超参数
- 优化器和调度器配置
- 特征配置

### 3. `modeling_pi05.py` - 核心模型实现

**功能**：实现 PI05 的完整模型架构和训练/推理逻辑

**主要组件**：
- 工具函数（位置编码、掩码生成等）
- `PaliGemmaWithExpertModel`：视觉-语言-动作模型
- `PI05Pytorch`：核心 PyTorch 模型
- `PI05Policy`：LeRobot 策略接口

### 4. `processor_pi05.py` - 数据处理器

**功能**：定义数据预处理和后处理管道

**主要组件**：
- `Pi05PrepareStateTokenizerProcessorStep`：状态和语言预处理
- `make_pi05_pre_post_processors`：创建预处理和后处理管道

### 5. `README.md` - 说明文档

**功能**：模型概述、特性对比、引用信息

---

## 代码块功能详解

### `__init__.py`

```python
from .configuration_pi05 import PI05Config
from .modeling_pi05 import PI05Policy
from .processor_pi05 import make_pi05_pre_post_processors

__all__ = ["PI05Config", "PI05Policy", "make_pi05_pre_post_processors"]
```

**功能**：导出模块的公共接口

---

### `configuration_pi05.py`

#### 1. `PI05Config` 类定义

**功能**：PI05 模型的配置类，继承自 `PreTrainedConfig`

**关键配置参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `paligemma_variant` | str | "gemma_2b" | PaliGemma 模型变体 |
| `action_expert_variant` | str | "gemma_300m" | Action Expert 模型变体 |
| `dtype` | str | "float32" | 数据类型 |
| `chunk_size` | int | 50 | 预测的动作步数 |
| `n_action_steps` | int | 50 | 执行的动作步数 |
| `max_state_dim` | int | 32 | 状态最大维度 |
| `max_action_dim` | int | 32 | 动作最大维度 |
| `num_inference_steps` | int | 10 | 推理时的去噪步数 |

#### 2. `__post_init__` 方法

**功能**：配置验证
- 验证 `n_action_steps <= chunk_size`
- 验证模型变体有效性
- 验证数据类型有效性

#### 3. `validate_features` 方法

**功能**：验证和设置输入/输出特征
- 添加空相机特征（如果需要）
- 设置状态特征（填充到 `max_state_dim`）
- 设置动作特征（填充到 `max_action_dim`）

#### 4. `get_optimizer_preset` 方法

**功能**：返回优化器配置（AdamW）

#### 5. `get_scheduler_preset` 方法

**功能**：返回调度器配置（余弦衰减 + warmup）

#### 6. `action_delta_indices` 属性

**功能**：返回动作增量索引（支持增量动作）

---

### `processor_pi05.py`

#### 1. `Pi05PrepareStateTokenizerProcessorStep` 类

**功能**：准备状态和语言输入，用于分词

**处理步骤**：
1. 获取状态和任务描述
2. 将状态填充到 `max_state_dim`
3. 将状态离散化为 256 个 bins
4. 构建完整提示："Task: {task}, State: {state};\nAction: "

**关键变量**：
- `state`：机器人状态（归一化到 [-1, 1]）
- `discretized_states`：离散化后的状态（0-255）
- `full_prompts`：包含任务和状态的完整提示

#### 2. `make_pi05_pre_post_processors` 函数

**功能**：创建预处理和后处理管道

**预处理管道步骤**：
1. `RenameObservationsProcessorStep`：重命名特征
2. `AddBatchDimensionProcessorStep`：添加批维度
3. `NormalizerProcessorStep`：归一化（状态和动作使用分位数归一化）
4. `Pi05PrepareStateTokenizerProcessorStep`：准备状态和语言
5. `TokenizerProcessorStep`：使用 PaliGemma tokenizer 分词
6. `DeviceProcessorStep`：移到指定设备

**后处理管道步骤**：
1. `UnnormalizerProcessorStep`：反归一化动作
2. `DeviceProcessorStep`：移到 CPU

---

### `modeling_pi05.py`

#### 一、工具函数

##### 1. `get_safe_dtype(target_dtype, device_type)`

**功能**：根据设备类型返回安全的数据类型

**参数**：
- `target_dtype`：目标数据类型
- `device_type`：设备类型（"cpu", "cuda", "mps"）

**返回值**：安全的数据类型

**处理规则**：
- MPS 设备：`float64` → `float32`
- CPU 设备：`bfloat16` → `float32`
- 其他情况：返回原类型

##### 2. `create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device)`

**功能**：创建正弦-余弦位置编码（用于时间嵌入）

**参数**：
- `time`：时间标量 `[B]`
- `dimension`：编码维度（必须是偶数）
- `min_period`：最小周期（默认 4e-3）
- `max_period`：最大周期（默认 4.0）
- `device`：设备

**返回值**：位置编码向量 `[B, dimension]`

**公式**：
```
period = min_period * (max_period / min_period) ^ fraction
scaling = 1.0 / period * 2π
embedding = [sin(scaling * time), cos(scaling * time)]
```

##### 3. `sample_beta(alpha, beta, bsize, device)`

**功能**：从 Beta 分布采样时间步

**参数**：
- `alpha`：Beta 分布参数 α（默认 1.5）
- `beta`：Beta 分布参数 β（默认 1.0）
- `bsize`：批次大小
- `device`：设备

**返回值**：采样值 `[B]`，范围 [0, 1]

**用途**：训练时的时间采样策略

##### 4. `make_att_2d_masks(pad_masks, att_masks)`

**功能**：构建 2D 注意力掩码

**参数**：
- `pad_masks`：填充掩码 `[B, N]`，`True`=有效位置
- `att_masks`：注意力掩码 `[B, N]`，控制注意力模式

**返回值**：2D 注意力掩码 `[B, N, N]`，布尔类型

**支持的注意力模式**：
- 因果注意力：`[1, 1, 1, 1, 1, 1]`
- 前缀 LM：`[0, 0, 0, 1, 1, 1]`
- 块级因果：`[1, 0, 1, 0, 1, 0, 0, 1, 0, 0]`

##### 5. `pad_vector(vector, new_dim)`

**功能**：将向量最后一维填充到指定维度

**参数**：
- `vector`：输入向量，形状 `[B, D]` 或 `[B, L, D]`
- `new_dim`：目标维度

**返回值**：填充后的向量，最后一维为 `new_dim`

**处理**：如果 `vector.shape[-1] >= new_dim`，直接返回；否则用零填充

##### 6. `resize_with_pad_torch(images, height, width, mode)`

**功能**：图像缩放与填充（保持宽高比）

**参数**：
- `images`：图像张量，`[B, C, H, W]` 或 `[B, H, W, C]`
- `height`：目标高度
- `width`：目标宽度
- `mode`：插值模式（"bilinear", "nearest" 等）

**返回值**：缩放和填充后的图像，保持原始格式

**处理**：
1. 计算缩放比例（保持宽高比）
2. 缩放图像
3. 用黑色填充到目标尺寸
4. `uint8` 类型：填充值为 0
5. `float32` 类型：填充值为 -1.0（SigLIP 要求）

##### 7. `compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert)`

**功能**：用于梯度检查点的完整层计算

**参数**：
- `layer_idx`：层索引
- `inputs_embeds`：输入嵌入列表 `[prefix_embs, suffix_embs]`
- `attention_mask`：注意力掩码
- `position_ids`：位置 ID
- `adarms_cond`：AdaRMS 条件列表 `[None, time_emb]`
- `paligemma`：PaliGemma 模型
- `gemma_expert`：Gemma Expert 模型

**返回值**：输出嵌入列表 `[prefix_out, suffix_out]`

**处理步骤**：
1. 对 prefix 和 suffix 分别进行 Input LayerNorm（suffix 使用 AdaRMS）
2. 分别计算 QKV
3. 拼接 QKV 并应用 RoPE
4. 联合注意力计算
5. 分别进行输出投影
6. 第一个残差连接
7. Post-attention LayerNorm（suffix 使用 AdaRMS）
8. MLP 处理
9. 第二个残差连接

#### 二、配置相关

##### 8. `GemmaConfig` 类

**功能**：Gemma 模型配置容器
- 属性：`width`, `depth`, `mlp_dim`, `num_heads`, `num_kv_heads`, `head_dim`

##### 9. `get_gemma_config(variant: str)`

**功能**：根据变体返回配置
- 支持：`gemma_300m`（1024 隐藏维，18 层）和 `gemma_2b`（2048 隐藏维，18 层）

#### 三、模型类

##### 10. `PaliGemmaWithExpertModel` 类

**功能**：结合视觉-语言模型与动作专家模型

**架构组成**：
- `self.paligemma`：PaliGemma 模型（处理图像+语言）
- `self.gemma_expert`：Gemma Expert 模型（处理动作）

**关键方法**：

###### `__init__(vlm_config, action_expert_config, use_adarms, precision)`

**功能**：初始化 PaliGemma 和 Gemma Expert

**参数**：
- `vlm_config`：视觉-语言模型配置（GemmaConfig）
- `action_expert_config`：动作专家模型配置（GemmaConfig）
- `use_adarms`：AdaRMS 使用列表 `[paligemma_use, expert_use]`
- `precision`：精度（"bfloat16" 或 "float32"）

**初始化**：
1. 配置 PaliGemma（使用 HuggingFace 配置）
2. 配置 Gemma Expert（启用 AdaRMS）
3. 移除 Expert 的 token embedding（使用 `inputs_embeds`）
4. 应用精度转换

###### `to_bfloat16_for_selected_params(precision)`

**功能**：将部分参数转为 `bfloat16`，保留关键层为 `float32`

**保留 `float32` 的参数**：
- 视觉嵌入层（patch_embedding, position_embedding）
- LayerNorm 层（input_layernorm, post_attention_layernorm）
- 模型归一化层（model.norm）

**原因**：这些层对数值精度敏感，保持 `float32` 有助于训练稳定性

###### `embed_image(image)`

**功能**：使用 SigLIP 编码图像

**输入**：图像 `[B, C, H, W]`，归一化到 `[-1, 1]`

**返回值**：图像嵌入 `[B, num_patches, hidden_dim]`

**实现**：`self.paligemma.model.get_image_features(image)`

###### `embed_language_tokens(tokens)`

**功能**：编码语言 token

**输入**：Token IDs `[B, seq_len]`

**返回值**：Token 嵌入 `[B, seq_len, hidden_dim]`

**实现**：`self.paligemma.language_model.embed_tokens(tokens)`

###### `forward(attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond)`

**功能**：前向传播，支持三种模式

**参数**：
- `attention_mask`：注意力掩码 `[B, N, N]`
- `position_ids`：位置 ID `[B, N]`
- `past_key_values`：缓存的 KV（推理时使用）
- `inputs_embeds`：输入嵌入列表 `[prefix_embs, suffix_embs]`
- `use_cache`：是否使用 KV cache
- `adarms_cond`：AdaRMS 条件列表 `[paligemma_cond, expert_cond]`

**返回值**：`([prefix_output, suffix_output], past_key_values)`

**三种模式**：

1. **仅 Prefix**（`inputs_embeds[1] is None`）：
   - 只处理图像+语言
   - 返回 `prefix_output` 和 `past_key_values`
   - 用于推理时缓存 Prefix KV

2. **仅 Suffix**（`inputs_embeds[0] is None`）：
   - 只处理动作
   - 返回 `suffix_output`
   - 用于推理时的去噪步骤

3. **联合处理**（两者都不为 None）：
   - 同时处理 prefix 和 suffix
   - 通过 `compute_layer_complete` 逐层处理
   - 支持梯度检查点
   - 返回 `[prefix_output, suffix_output]`
   - 用于训练时

##### 11. `PI05Pytorch` 类

**功能**：核心 PI05 PyTorch 模型

**关键方法**：

###### `__init__(config, rtc_processor)`

**功能**：初始化模型组件

**创建的网络层**：
- `paligemma_with_expert`：PaliGemma + Gemma Expert 联合模型
- `action_in_proj`：动作输入投影 `[max_action_dim → hidden_dim]`
- `action_out_proj`：动作输出投影 `[hidden_dim → max_action_dim]`（动作头）
- `time_mlp_in`：时间 MLP 输入层
- `time_mlp_out`：时间 MLP 输出层

**可选功能**：
- 梯度检查点（如果启用）
- 模型编译（如果启用）

###### `sample_noise(shape, device)`

**功能**：采样标准正态噪声

**参数**：
- `shape`：噪声形状
- `device`：设备

**返回值**：噪声张量 `~N(0, 1)`，`dtype=float32`

###### `sample_time(bsize, device)`

**功能**：采样训练时间步（Beta 分布）

**参数**：
- `bsize`：批次大小
- `device`：设备

**返回值**：时间步 `[B]`，范围 `[offset, offset+scale]`

**公式**：
```
time_beta ~ Beta(alpha, beta)
time = time_beta * scale + offset
```

###### `embed_prefix(images, img_masks, tokens, masks)`

**功能**：编码前缀（图像 + 语言）

**参数**：
- `images`：图像列表
- `img_masks`：图像掩码列表
- `tokens`：语言 token `[B, seq_len]`
- `masks`：语言 token 掩码 `[B, seq_len]`

**返回值**：`(embs, pad_masks, att_masks)`
- `embs`：拼接后的嵌入 `[B, total_len, hidden_dim]`
- `pad_masks`：填充掩码 `[B, total_len]`
- `att_masks`：注意力掩码 `[B, total_len]`

**处理**：
1. 图像通过 SigLIP 编码
2. 语言 token 通过 embedding 层编码，并乘以 `√(embed_dim)`
3. 拼接图像和语言嵌入
4. 生成相应的掩码

###### `embed_suffix(noisy_actions, timestep)`

**功能**：编码后缀（带噪声动作 + 时间步）

**参数**：
- `noisy_actions`：带噪声的动作 `[B, chunk_size, max_action_dim]`
- `timestep`：时间步 `[B]`

**返回值**：`(embs, pad_masks, att_masks, adarms_cond)`
- `embs`：动作嵌入 `[B, chunk_size, hidden_dim]`
- `pad_masks`：填充掩码 `[B, chunk_size]`
- `att_masks`：注意力掩码 `[B, chunk_size]`
- `adarms_cond`：AdaRMS 条件（时间嵌入）`[B, hidden_dim]`

**处理**：
1. 时间步通过正弦-余弦位置编码
2. 时间嵌入通过时间 MLP 处理
3. 动作通过 `action_in_proj` 投影
4. 时间嵌入作为 AdaRMS 条件

###### `forward(images, img_masks, tokens, masks, actions, noise=None, time=None)`

**功能**：训练前向，计算 Flow Matching 损失

**参数**：
- `images`：图像列表
- `img_masks`：图像掩码列表
- `tokens`：语言 token
- `masks`：语言 token 掩码
- `actions`：真实动作（ground truth）`[B, chunk_size, max_action_dim]`
- `noise`：可选，噪声（如果不提供则采样）
- `time`：可选，时间步（如果不提供则采样）

**返回值**：损失张量 `[B, chunk_size, max_action_dim]`（MSE，未求平均）

**流程**：
1. 采样噪声和时间（如果未提供）
2. 构建插值路径：`x_t = t * noise + (1-t) * actions`
3. 计算真实速度场：`u_t = noise - actions`
4. 编码 prefix 和 suffix
5. 通过 Transformer 处理
6. 预测速度场：`v_t = action_out_proj(suffix_out)`
7. 计算损失：`MSE(u_t, v_t)`

###### `sample_actions(images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs)`

**功能**：推理时采样动作（多步去噪）

**参数**：
- `images`：图像列表
- `img_masks`：图像掩码列表
- `tokens`：语言 token
- `masks`：语言 token 掩码
- `noise`：可选，初始噪声
- `num_steps`：可选，去噪步数（默认 10）
- `**kwargs`：RTC 相关参数

**返回值**：预测动作 `[B, chunk_size, max_action_dim]`

**流程**：
1. 初始化噪声：`x_t = noise`（t=1）
2. 处理 Prefix 并缓存 KV
3. 迭代去噪（从 t=1 到 t≈0）：
   - 预测速度场 `v_t`
   - Euler 积分步：`x_t += dt * v_t`
   - 更新时间：`t += dt`
4. 返回最终动作：`x_t`（t≈0）

###### `denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)`

**功能**：单步去噪

**参数**：
- `prefix_pad_masks`：Prefix 填充掩码
- `past_key_values`：缓存的 Prefix KV
- `x_t`：当前状态 `[B, chunk_size, max_action_dim]`
- `timestep`：当前时间步 `[B]`

**返回值**：速度场 `v_t` `[B, chunk_size, max_action_dim]`

**流程**：
1. 编码 suffix（`x_t` + `timestep`）
2. 构建完整注意力掩码
3. 通过 Transformer 处理（复用 Prefix KV）
4. 通过动作头预测速度场

##### 12. `PI05Policy` 类

**功能**：LeRobot 策略接口，继承自 `PreTrainedPolicy`

**关键方法**：

###### `__init__(config)`

**功能**：初始化策略

**流程**：
1. 调用父类初始化
2. 验证特征配置
3. 初始化 RTC 处理器（如果启用）
4. 创建 `PI05Pytorch` 模型
5. 启用梯度检查点（如果配置）
6. 将模型移到指定设备
7. 重置内部状态（动作队列）

###### `from_pretrained(...)`

**功能**：从预训练加载模型

**参数**：
- `pretrained_name_or_path`：预训练模型路径
- `config`：可选，配置对象
- 其他 HuggingFace 加载参数

**返回值**：加载后的 `PI05Policy` 实例

**处理**：
1. 显示免责声明（OpenPI 移植说明）
2. 加载配置
3. 初始化模型
4. 加载状态字典（支持 safetensors）
5. 修复状态字典键（处理架构差异）
6. 添加 "model." 前缀（如果需要）
7. 加载权重并报告缺失/意外键

###### `_fix_pytorch_state_dict_keys(state_dict, model_config)`

**功能**：修复状态字典键以匹配当前架构

**处理**：
- 处理 LayerNorm 结构变化（adaRMS 相关）
- 处理 MLP 命名变化（`action_time_mlp_*` → `time_mlp_*`）
- 跳过 `state_proj`（PI05 不使用）
- 处理视觉嵌入层差异

###### `_preprocess_images(batch)`

**功能**：预处理图像

**输入**：`batch` 字典，包含图像键

**返回值**：`(images, img_masks)`
- `images`：预处理后的图像列表
- `img_masks`：图像掩码列表

**处理**：
1. 确保设备一致
2. 转换为 `float32`
3. 处理通道顺序（支持 `[B, C, H, W]` 和 `[B, H, W, C]`）
4. 缩放和填充到 `image_resolution`（224x224）
5. 归一化：`[0, 1]` → `[-1, 1]`（SigLIP 要求）
6. 处理缺失图像（填充为 -1，掩码为 0）

###### `prepare_action(batch)`

**功能**：填充动作到 `max_action_dim`

**输入**：`batch` 字典，包含 `batch[ACTION]`

**返回值**：填充后的动作 `[B, chunk_size, max_action_dim]`

###### `select_action(batch)`

**功能**：选择单个动作（不支持 RTC）

**输入**：`batch` 字典

**返回值**：单个动作 `[B, action_dim]`

**处理**：
1. 如果动作队列为空，调用 `predict_action_chunk` 生成动作块
2. 将动作块加入队列
3. 从队列中取出一个动作返回

**注意**：不支持 RTC，如果启用 RTC 会抛出异常

###### `predict_action_chunk(batch, **kwargs)`

**功能**：预测动作块（支持 RTC）

**输入**：
- `batch`：观察数据字典
- `**kwargs`：RTC 相关参数（`inference_delay`, `prev_chunk_left_over`, `execution_horizon`）

**返回值**：动作块 `[B, chunk_size, action_dim]`

**处理**：
1. 预处理图像
2. 调用 `model.sample_actions` 生成动作
3. 去填充到实际动作维度

###### `forward(batch)`

**功能**：训练前向，计算损失

**输入**：`batch` 字典，包含 ground truth actions

**返回值**：`(loss, loss_dict)`
- `loss`：标量损失值
- `loss_dict`：包含 `loss` 和 `loss_per_dim` 的字典

**处理**：
1. 预处理图像
2. 准备动作（填充）
3. 调用 `model.forward` 计算损失
4. 截取到实际动作维度
5. 计算平均损失

---

## 关键代码块详细说明

### Flow Matching 核心实现

#### 训练时的 Flow Matching 损失计算

```690:743:piper_lerobot/src/lerobot/policies/pi05/modeling_pi05.py
    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")
```

**关键步骤说明**：

# 1. 采样噪声和时间
noise = self.sample_noise(actions.shape, actions.device)  # ~N(0, 1)
time = self.sample_time(actions.shape[0], actions.device)  # ~Beta(α, β)

# 2. 构建插值路径
time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions
# x_t 在噪声和真实动作之间插值
# t=0: x_0 = actions (真实动作)
# t=1: x_1 = noise (纯噪声)

# 3. 计算真实速度场
u_t = noise - actions
# 这是模型要学习预测的目标

# 4. 模型预测速度场
v_t = model(x_t, t, images, language)
# 通过 Transformer 预测速度场

# 5. 计算损失
loss = MSE(u_t, v_t)
# 让模型预测接近真实速度场
```

#### 推理时的迭代去噪

```787:830:piper_lerobot/src/lerobot/policies/pi05/modeling_pi05.py
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            # Define a closure function to properly capture expanded_time
            # This avoids the lambda expression (E731) and loop variable binding (B023) issues
            def denoise_step_partial_call(input_x_t, current_timestep=expanded_time):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            # Euler step
            x_t += dt * v_t

            # Record x_t and v_t after Euler step
            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

            time += dt

        return x_t
```

**关键步骤说明**：
1. **初始化**：`x_t = noise`（从纯噪声开始），`time = 1.0`（从 t=1 开始）
2. **时间步长**：`dt = -1.0 / num_steps`（负数，向后积分）
3. **迭代去噪**：从 t=1 到 t≈0，每次预测速度场并更新状态
4. **Euler 积分**：`x_t += dt * v_t`（沿着预测方向更新）
5. **终止条件**：`time >= -dt / 2`（确保覆盖到 t=0）

### 注意力机制实现

#### 联合注意力处理

```python
# 位置：compute_layer_complete (第 219-289 行)

# 1. 分别处理 prefix 和 suffix
for i, hidden_states in enumerate(inputs_embeds):
    # prefix (i=0) 和 suffix (i=1) 分别处理
    layer = models[i].layers[layer_idx]
    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
    # suffix 使用 AdaRMS 条件（时间嵌入）
    
    # 计算 QKV
    query_state = layer.self_attn.q_proj(hidden_states)
    key_state = layer.self_attn.k_proj(hidden_states)
    value_state = layer.self_attn.v_proj(hidden_states)

# 2. 拼接 QKV
query_states = torch.cat([prefix_q, suffix_q], dim=2)
key_states = torch.cat([prefix_k, suffix_k], dim=2)
value_states = torch.cat([prefix_v, suffix_v], dim=2)

# 3. 应用 RoPE
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# 4. 联合注意力计算
att_output = eager_attention_forward(query_states, key_states, value_states, attention_mask)

# 5. 分别投影回各自维度
prefix_out = prefix_layer.o_proj(att_output[:, :prefix_len])
suffix_out = suffix_layer.o_proj(att_output[:, prefix_len:])
```

### 状态离散化处理

```58:89:piper_lerobot/src/lerobot/policies/pi05/processor_pi05.py
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition
```

**关键步骤说明**：
1. **填充状态**：`pad_vector(state, max_state_dim)` 确保状态维度一致
2. **离散化**：将归一化到 `[-1, 1]` 的状态值映射到 `[0, 255]` 的整数
3. **构建提示**：将任务和离散化状态组合成文本提示，供 tokenizer 处理

### 动作队列管理

```python
# 位置：PI05Policy.select_action (第 1174-1189 行)

# 动作队列用于管理多步动作
self._action_queue = deque(maxlen=n_action_steps)

# 当队列为空时，生成新的动作块
if len(self._action_queue) == 0:
    actions = self.predict_action_chunk(batch)[:, :n_action_steps]
    # 形状：[B, n_action_steps, action_dim]
    
    # 转置并加入队列
    self._action_queue.extend(actions.transpose(0, 1))
    # 队列中每个元素：[B, action_dim]

# 从队列中取出一个动作
return self._action_queue.popleft()
```

---

## 变量含义

### 核心变量

#### 训练相关

| 变量 | 类型 | 含义 |
|------|------|------|
| `x_t` | `Tensor` | Flow Matching 插值路径上的点：`x_t = t * noise + (1-t) * actions` |
| `u_t` | `Tensor` | 真实速度场：`u_t = noise - actions`（训练时的目标） |
| `v_t` | `Tensor` | 模型预测的速度场：`v_t = model(x_t, t)` |
| `noise` | `Tensor` | 标准正态噪声：`~N(0, 1)` |
| `time` / `t` | `Tensor` | Flow Matching 时间步：`t ∈ [0, 1]` |
| `actions` | `Tensor` | 真实动作（ground truth） |

#### 模型相关

| 变量 | 类型 | 含义 |
|------|------|------|
| `prefix_embs` | `Tensor` | Prefix 嵌入（图像+语言）：`[B, prefix_len, hidden_dim]` |
| `suffix_embs` | `Tensor` | Suffix 嵌入（动作+时间）：`[B, suffix_len, hidden_dim]` |
| `suffix_out` | `Tensor` | Action Expert 输出的隐式动作：`[B, chunk_size, hidden_dim]` |
| `past_key_values` | `list` | KV cache，用于推理时缓存 Prefix 的注意力 |

#### 掩码相关

| 变量 | 类型 | 含义 |
|------|------|------|
| `pad_masks` | `Tensor` | 填充掩码：`True`=有效位置，`False`=填充位置 |
| `att_masks` | `Tensor` | 注意力掩码：控制注意力模式（因果、前缀等） |
| `att_2d_masks` | `Tensor` | 2D 注意力掩码：`[B, N, N]`，用于 Transformer |

#### 配置相关

| 变量 | 类型 | 含义 |
|------|------|------|
| `chunk_size` | `int` | 预测的动作块大小（默认 50） |
| `n_action_steps` | `int` | 执行的动作步数（默认 50） |
| `max_action_dim` | `int` | 动作最大维度（默认 32） |
| `max_state_dim` | `int` | 状态最大维度（默认 32） |
| `num_inference_steps` | `int` | 推理时的去噪步数（默认 10） |

---

## 调用顺序

### 训练时的调用顺序

```
1. 数据加载
   └─> batch[ACTION] = ground_truth_actions

2. 预处理管道（preprocessor）
   ├─> RenameObservationsProcessorStep
   ├─> AddBatchDimensionProcessorStep
   ├─> NormalizerProcessorStep  # 归一化状态和动作
   ├─> Pi05PrepareStateTokenizerProcessorStep
   │   ├─> pad_vector(state, max_state_dim)
   │   ├─> 离散化状态为 256 bins
   │   └─> 构建提示："Task: {task}, State: {state};\nAction: "
   ├─> TokenizerProcessorStep  # PaliGemma tokenizer
   └─> DeviceProcessorStep

3. PI05Policy.forward(batch)
   ├─> _preprocess_images(batch)  # 预处理图像
   ├─> prepare_action(batch)  # 填充动作
   │   └─> pad_vector(batch[ACTION], max_action_dim)
   └─> model.forward(images, img_masks, tokens, masks, actions)

4. PI05Pytorch.forward(...)
   ├─> sample_noise(actions.shape)  # 采样噪声
   ├─> sample_time(bsize)  # 采样时间步
   ├─> x_t = t * noise + (1-t) * actions  # 构建插值路径
   ├─> u_t = noise - actions  # 真实速度场
   ├─> embed_prefix(images, img_masks, tokens, masks)
   │   ├─> paligemma_with_expert.embed_image(img)  # SigLIP 编码
   │   └─> paligemma_with_expert.embed_language_tokens(tokens)  # Token 嵌入
   ├─> embed_suffix(x_t, time)
   │   ├─> create_sinusoidal_pos_embedding(time, ...)  # 时间嵌入
   │   ├─> action_in_proj(x_t)  # 动作投影
   │   └─> time_mlp(time_emb)  # 时间 MLP
   ├─> paligemma_with_expert.forward([prefix_embs, suffix_embs], ...)
   │   └─> compute_layer_complete(...)  # 逐层处理（18层）
   │       ├─> Input LayerNorm
   │       ├─> QKV 投影
   │       ├─> 拼接 + RoPE
   │       ├─> 联合注意力
   │       ├─> 输出投影
   │       ├─> 残差连接
   │       ├─> Post LayerNorm
   │       ├─> MLP
   │       └─> 残差连接
   ├─> suffix_out = outputs[1]  # 隐式动作
   ├─> suffix_out[:, -chunk_size:]  # 取最后 chunk_size 个 token
   ├─> action_out_proj(suffix_out)  # 动作头 → v_t
   └─> MSE(u_t, v_t)  # 计算损失

5. 反向传播
   └─> 更新模型参数
```

### 推理时的调用顺序

```
1. 观察输入
   └─> batch = {images, state, task, ...}

2. 预处理管道（preprocessor）
   └─> （同训练时）

3. PI05Policy.predict_action_chunk(batch)
   ├─> _preprocess_images(batch)
   └─> model.sample_actions(images, img_masks, tokens, masks)

4. PI05Pytorch.sample_actions(...)
   ├─> sample_noise(...)  # 初始化噪声
   ├─> embed_prefix(...)  # 编码图像+语言
   ├─> paligemma_with_expert.forward([prefix_embs, None], use_cache=True)
   │   └─> 缓存 past_key_values
   ├─> x_t = noise  # 初始化
   ├─> time = 1.0
   └─> while time >= -dt / 2:  # 迭代去噪（10步）
       ├─> denoise_step(x_t, time)
       │   ├─> embed_suffix(x_t, time)
       │   ├─> paligemma_with_expert.forward([None, suffix_embs], past_key_values)
       │   │   └─> 复用缓存的 Prefix KV
       │   └─> action_out_proj(suffix_out)  # 预测 v_t
       ├─> x_t += dt * v_t  # Euler 积分步
       └─> time += dt
   └─> return x_t  # 最终动作

5. 后处理管道（postprocessor）
   ├─> UnnormalizerProcessorStep  # 反归一化
   └─> DeviceProcessorStep  # 移到 CPU

6. 环境执行
   └─> env.step(action)  # 发送到机器人/仿真器
```

---

## 数据流

### 训练数据流

```
数据集
  ↓
EnvTransition (包含 action, state, images, task)
  ↓
预处理管道
  ├─> 归一化 action, state
  ├─> 状态离散化
  ├─> 构建提示文本
  └─> 分词
  ↓
Batch 字典
  ├─> batch[ACTION]: [B, action_dim] (ground truth)
  ├─> batch[OBS_STATE]: [B, state_dim]
  ├─> batch[OBS_LANGUAGE_TOKENS]: [B, seq_len]
  └─> batch[observation.images.*]: [B, C, H, W]
  ↓
PI05Policy.forward(batch)
  ├─> 预处理图像
  ├─> 填充动作到 max_action_dim
  └─> 模型前向
      ├─> Flow Matching 插值: x_t = t*noise + (1-t)*actions
      ├─> 真实速度场: u_t = noise - actions
      ├─> 模型预测: v_t = model(x_t, t, images, language)
      └─> 损失: MSE(u_t, v_t)
```

### 推理数据流

```
观察输入
  ├─> 图像: [B, C, H, W]
  ├─> 状态: [B, state_dim]
  └─> 任务: str
  ↓
预处理管道
  └─> （同训练时）
  ↓
PI05Policy.predict_action_chunk(batch)
  ↓
PI05Pytorch.sample_actions(...)
  ├─> 处理 Prefix（图像+语言）→ 缓存 KV
  └─> 迭代去噪 Suffix（动作）
      ├─> t=1.0: x_t = noise
      ├─> t=0.9: x_t += dt * v_t
      ├─> ...
      └─> t≈0: x_t = 预测动作
  ↓
后处理管道
  ├─> 反归一化
  └─> 移到 CPU
  ↓
动作输出: [B, chunk_size, action_dim]
  ↓
env.step(action)  # 控制执行
```

---

## 关键概念说明

### Prefix 和 Suffix

- **Prefix（前缀）**：包含观察信息（图像 + 语言）
  - 使用 PaliGemma 的 `language_model` 处理
  - 在推理时可以缓存 KV 以提高效率

- **Suffix（后缀）**：包含动作信息（带噪声动作 + 时间步）
  - 使用 Gemma Expert 处理
  - 使用 AdaRMS 条件（时间嵌入）

### Flow Matching

- **训练**：学习预测速度场 `v_t`，使其接近真实速度场 `u_t = noise - actions`
- **推理**：从噪声开始，通过迭代去噪生成动作

### 隐式动作 vs 显式动作

- **隐式动作**：`suffix_out`（Transformer 隐藏状态，维度：`hidden_dim`）
- **显式动作**：`v_t = action_out_proj(suffix_out)`（动作空间，维度：`action_dim`）

### Action 格式

- PI05 模型本身不限定 action 的物理含义
- 可以是关节角度、末端执行器位置、速度等
- 具体格式取决于数据集和任务配置
- 常见：ALOHA 使用 14 维关节角度

---

## 文件依赖关系

```
__init__.py
  ├─> configuration_pi05.py (PI05Config)
  ├─> modeling_pi05.py (PI05Policy)
  └─> processor_pi05.py (make_pi05_pre_post_processors)

modeling_pi05.py
  ├─> configuration_pi05.py (PI05Config)
  └─> processor_pi05.py (pad_vector)

processor_pi05.py
  └─> modeling_pi05.py (pad_vector)
```

---

## 架构图

### 模型架构概览

```
输入层
├─> 图像 → SigLIP 编码 → [Image Patches]
├─> 语言 → Token Embedding → [Language Tokens]
└─> 状态 → 离散化 → 文本提示 → Token Embedding

    ↓ 拼接

Prefix: [Image Patches] + [Language Tokens]
    ↓
PaliGemma Language Model (18层)
    ↓
Prefix 隐藏表示

    ↓ 联合注意力

Suffix: [Noisy Actions] + [Time Embedding]
    ↓
Gemma Expert (18层，AdaRMS 条件)
    ↓
Suffix 隐藏表示 (隐式动作)

    ↓
Action Head (线性层)
    ↓
显式动作 / 速度场
```

### 训练时的数据流

```
Ground Truth Actions [B, chunk_size, action_dim]
    ↓
采样噪声 noise ~ N(0,1)
采样时间 t ~ Beta(α,β)
    ↓
x_t = t * noise + (1-t) * actions  (插值路径)
u_t = noise - actions  (真实速度场)
    ↓
模型预测: v_t = model(x_t, t, images, language)
    ↓
损失: MSE(u_t, v_t)
```

### 推理时的数据流

```
观察 (图像 + 状态 + 任务)
    ↓
预处理 → Token 化
    ↓
Prefix 处理 → 缓存 KV
    ↓
初始化: x_t = noise, t = 1.0
    ↓
迭代去噪 (10步):
    ├─> 预测 v_t = model(x_t, t)
    ├─> 更新 x_t += dt * v_t
    └─> 更新 t += dt
    ↓
最终动作: x_t (t≈0)
    ↓
后处理 → 反归一化
    ↓
env.step(action) → 机器人控制
```

---

## 组件交互图

### 训练时的组件交互

```
DataLoader
  ↓ batch
Preprocessor Pipeline
  ├─> NormalizerProcessorStep
  ├─> Pi05PrepareStateTokenizerProcessorStep
  └─> TokenizerProcessorStep
  ↓ processed_batch
PI05Policy.forward()
  ├─> _preprocess_images()
  ├─> prepare_action()
  └─> PI05Pytorch.forward()
      ├─> embed_prefix() → PaliGemma
      ├─> embed_suffix() → Action Expert
      ├─> PaliGemmaWithExpertModel.forward()
      │   └─> compute_layer_complete() × 18层
      └─> action_out_proj() → v_t
  ↓ loss
Optimizer.step()
```

### 推理时的组件交互

```
Environment Observation
  ↓
Preprocessor Pipeline
  ↓
PI05Policy.predict_action_chunk()
  └─> PI05Pytorch.sample_actions()
      ├─> embed_prefix() → 缓存 KV
      └─> 迭代去噪循环:
          ├─> denoise_step()
          │   ├─> embed_suffix()
          │   ├─> PaliGemmaWithExpertModel.forward()
          │   │   └─> 复用 cached KV
          │   └─> action_out_proj()
          └─> Euler 积分更新
      ↓ actions
Postprocessor Pipeline
  └─> UnnormalizerProcessorStep
  ↓
env.step(action) → Robot Control
```

---

## 总结

PI05 是一个基于 Flow Matching 的视觉-语言-动作模型，通过以下关键组件实现：

1. **配置系统**：`PI05Config` 管理所有超参数
2. **数据处理**：预处理和后处理管道处理多模态输入
3. **模型架构**：PaliGemma + Gemma Expert 联合处理
4. **训练**：Flow Matching 损失（MSE(u_t, v_t)）
5. **推理**：迭代去噪生成动作序列

### 关键特性

- **多模态融合**：图像、语言、状态统一处理
- **高效推理**：Prefix KV 缓存，避免重复计算
- **灵活配置**：支持不同机器人平台和任务
- **Flow Matching**：基于连续流的动作生成

整个系统设计灵活，支持不同的机器人平台和任务配置。
