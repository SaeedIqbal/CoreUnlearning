# CoreUnlearning

## 项目概述
本项目聚焦于机器学习中的反学习（Unlearning）问题，实现了多种反学习算法，包括近似反学习（Approximate Unlearning）、核心反学习（Core Unlearning）、分布式反学习（Distributed Unlearning）、精确反学习（Exact Unlearning）、联邦反学习（Federated Unlearning）和图反学习（Graph Unlearning）。同时，提供了多个数据集的加载功能，支持不同类型数据的处理。

## 代码结构

### 主要文件及功能概述
| 文件名称 | 功能描述 |
| --- | --- |
| `au.py` | 实现近似反学习算法，通过多种技术优化模型权重以实现数据遗忘。 |
| `cu.py` | 实现核心反学习框架，整合多种学习模块处理复杂数据删除场景。 |
| `du.py` | 实现分布式反学习算法，在分布式环境中进行数据遗忘和模型更新。 |
| `eu.py` | 实现精确反学习算法，通过计算特定损失函数精确更新模型权重。 |
| `fu.py` | 实现联邦反学习算法，适用于联邦学习环境下的数据隐私保护。 |
| `gu.py` | 实现图反学习算法，处理图结构数据的反学习任务。 |
| `load_data.py` | 提供多个数据集的加载功能，包括图像、文本和图结构数据。 |

### 详细代码解释

#### `au.py`
- **类和函数**
    - **`PBRA` 类**：投影基残差调整（Projection - Based Residual Adjustment）。
        - `__init__`：初始化超参数 `eta`、`lambda_` 和 `gamma`。
        - `compute_loss`：计算 PBRA 损失，包括投影损失、权重正则化损失和投影张量正则化损失。
        - `update_weights`：根据 PBRA 方法更新权重。
    - **`ADBT` 类**：自适应决策边界调整（Adaptive Decision Boundary Tuning）。
        - `__init__`：初始化超参数 `alpha` 和 `lambda_`。
        - `compute_loss`：计算 ADBT 损失，考虑遗忘数据的熵和保留数据的梯度范数。
    - **`IBGP` 类**：影响基梯度修剪（Influence - Based Gradient Pruning）。
        - `__init__`：初始化超参数 `gamma`、`beta` 和 `lambda_`。
        - `compute_loss`：计算 IBGP 损失，通过门控概率调整遗忘数据的影响。
    - **`ApproximateUnlearning` 类**：
        - `__init__`：初始化 PBRA、ADBT 和 IBGP 类实例以及超参数 `zeta`。
        - `optimize_weights`：优化权重，依次执行 PBRA、ADBT 和 IBGP 步骤，并计算统一损失。

#### `cu.py`
- **类和函数**
    - **`CoreUnlearning` 类**：
        - `__init__`：初始化数据集、超参数和全局权重。
        - `compute_shard_specific_dynamics`：计算分片特定动态（SSD）。
        - `multi_modal_joint_optimization`：执行多模态联合优化。
        - `temporal_forgetting`：应用时间遗忘机制。
        - `hierarchical_forgetting`：实现分层遗忘机制。
        - `regularize_model`：对模型进行正则化以提高泛化能力。
        - `data_driven_shard_weighting`：进行数据驱动的分片加权。
        - `attention_mechanisms_for_unlearning`：应用注意力机制进行反学习。
        - `mixed_norm_regularization`：应用混合范数正则化进行多尺度反学习。
        - `unified_unlearning`：计算统一的反学习框架总损失。
        - `update_model`：根据总损失更新模型参数。

#### `du.py`
- **类和函数**
    - **`DistributedUnlearning` 类**：
        - `__init__`：初始化数据分片、学习率、超参数、反馈集大小和训练点数量。
        - `initialize_weights`：为每个数据分片初始化权重。
        - `calculate_loss_gradient`：计算给定数据集的损失梯度。
        - `update_weights`：根据梯度和学习率更新权重。
        - `compute_active_loss`：计算主动损失分量。
        - `incremental_update`：执行增量权重更新。
        - `distributed_loss`：计算所有数据分片的分布式损失。
        - `train`：使用分布式反学习算法训练模型。

#### `eu.py`
- **类和函数**
    - **`ExactUnlearning` 类**：
        - `__init__`：初始化模型和超参数。
        - `compute_loss_forget`：计算遗忘损失，包括对数项、梯度范数项和 Hessian 范数项。
        - `compute_loss_apa`：计算适配器分区和聚合（APA）损失。
        - `compute_loss_exact`：计算整体精确反学习损失。
        - `update_weights`：使用精确反学习方法更新模型权重。
        - `compute_hessian_inverse`：计算 Hessian 矩阵的逆（当前为占位符，需实际实现）。

#### `fu.py`
由于代码未提供，推测可能包含以下类和函数：
- **`FederatedUnlearning` 类**：
    - `__init__`：初始化联邦学习环境相关参数，如客户端数量、全局模型等。
    - `compute_local_loss`：计算客户端本地损失。
    - `aggregate_gradients`：聚合客户端梯度以更新全局模型。
    - `unlearn_data`：在联邦环境中实现数据遗忘。

#### `gu.py`
由于代码未提供，推测可能包含以下类和函数：
- **`GraphUnlearning` 类**：
    - `__init__`：初始化图数据、图神经网络模型和超参数。
    - `compute_graph_loss`：计算图数据的损失，考虑节点和边的信息。
    - `update_graph_weights`：更新图神经网络的权重以实现图数据的反学习。

#### `load_data.py`
- **函数**
    - **`load_image_dataset`**：通用的图像数据集加载函数，使用 `torchvision` 加载图像数据。
    - **`load_cifar10`**：加载 CIFAR - 10 数据集。
    - **`load_cifar100`**：加载 CIFAR - 100 数据集。
    - **`load_imdb4k`**：加载 IMDB4K 文本数据集，进行分词和词汇表构建。
    - **`load_cora`**：加载 Cora 图数据集。
    - **`load_femnist`**：加载 FEMNIST 数据集。
    - **`load_mvtec_ad`**：加载 MVTec AD 数据集，处理图像分类任务。

### 运行代码
1. 确保安装了所有必要的库，如 `torch`、`torchvision`、`torchtext`、`torch_geometric` 等。
2. 根据需求修改各文件中的超参数和数据集选择。
3. 运行主程序文件（如 `au.py`、`cu.py` 等），开始执行相应的反学习算法。

## 数据集引用

### CIFAR - 10 数据集
- **描述**：CIFAR - 10 数据集包含 60000 张 32x32 彩色图像，分为 10 个类别，每个类别有 6000 张图像。
- **引用**：[CIFAR - 10 and CIFAR - 100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

### CIFAR - 100 数据集
- **描述**：CIFAR - 100 数据集包含 60000 张 32x32 彩色图像，分为 100 个类别，每个类别有 600 张图像。
- **引用**：[CIFAR - 10 and CIFAR - 100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

### IMDB4K 数据集
- **描述**：IMDB4K 是一个文本数据集，包含电影评论数据，用于情感分析任务。
- **引用**：[IMDB Reviews](https://pytorch.org/text/stable/datasets.html#imdb)

### Cora 数据集
- **描述**：Cora 是一个图数据集，常用于图神经网络的研究。
- **引用**：[Planetoid: A Lightweight Library for Benchmarking Graph Neural Networks](https://pytorch - geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid)

### FEMNIST 数据集
- **描述**：FEMNIST 是一个手写数字和字符的图像数据集。
- **引用**：[Federated Extended MNIST (FEMNIST) Dataset](https://pytorch.org/vision/stable/datasets.html#femnist)

### MVTec AD 数据集
- **描述**：MVTec AD 是一个用于异常检测的图像数据集，包含多个类别的工业产品图像。
- **引用**：[MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec - ad)

## 依赖库
- `torch`
- `torchvision`
- `torchtext`
- `torch_geometric`

## 许可证
本项目遵循 [MIT 许可证](https://opensource.org/licenses/MIT)，你可以自由使用、修改和分发本项目的代码。