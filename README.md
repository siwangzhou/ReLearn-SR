# ReLearn-SR

本项目旨在通过再学习（ReLearn）的机制，对图像超分辨率（Super-Resolution, SR）的结果进行优化。代码库包含了一个测试脚本 `codes/test_relearn.py`，用于评估和优化预训练的SR模型在指定数据集上的表现。

## 功能特性

- **模型加载与配置**：支持从配置文件（如 `.yml` 文件）加载模型和测试参数。
- **数据集处理**：能够创建和加载测试数据集及对应的数据加载器。
- **迭代优化**：
    - 对输入的低分辨率（LR）图像添加一个可学习的扰动 `delta`。
    - 通过多轮迭代优化 `delta`，使得经过SR模型处理后的高分辨率（HR）图像在特定损失函数下表现更优。
    - 使用 `Adam` 优化器和 `CosineAnnealingLR` 学习率调度器。
- **图像质量评估**：
    - 计算多种图像质量指标，包括 PSNR、SSIM、PSNR-Y、SSIM-Y 和 LPIPS。
    - 记录迭代过程中的初始指标、最佳指标以及指标的提升/降低幅度。
- **结果保存**：
    - 保存优化过程中的关键图像，包括：
        - 初始生成的SR图像 (`_first.png`)
        - 优化后达到最佳指标的SR图像 (`_best.png`)
        - 最佳扰动 `delta` 的可视化图像 (`_delta_best_view.png` 和 `_delta_best.png`)
        - 最佳扰动 `delta` 经过上采样后的图像 (`_delta_best_sr.png`)
        - 初始的LR图像 (`_first_lr.png`)
        - 通过双三次插值下采样HR得到的LR图像 (`_lr_bic.png`)
    - 将每张图像的详细测试结果以及所有图像的平均结果保存到格式化的 Excel 文件中。
    - Excel 文件中包含GPU信息、模型名称以及各项指标的初始值、最值和变化量。
- **日志记录**：详细记录测试过程中的信息，包括配置、数据集信息、每张图像的评估结果和耗时。

## 使用说明

1.  **配置环境**：
    确保已安装所需的 Python 依赖库，例如 `torch`, `lpips`, `pandas`, `openpyxl`, `tqdm` 等。

2.  **准备数据集和模型**：
    -   根据 `options` 目录下的配置文件模板，准备好你的测试配置文件，指定数据集路径、预训练模型路径等。
    -   确保数据集和预训练模型文件已放置在正确的路径。

3.  **运行测试脚本**：
    可以通过命令行运行 `codes/test_relearn.py` 脚本。默认情况下，它会加载 `options/test/test_P2P_HCD_CARN_conv_4X.yml` 作为配置文件。
    ```bash
    python codes/test_relearn.py --opt path/to/your/config.yml
    ```
    将 `path/to/your/config.yml` 替换为你的配置文件路径。

4.  **查看结果**：
    -   测试完成后，生成的图像和 Excel 结果文件将保存在 `results/<test_set_name>/<image_name>/` 和 `results/<test_set_name>/` 目录下。
    -   日志信息会输出到屏幕并保存到 `experiments/test_results/<test_set_name>/log/` 目录下的日志文件中。

## 代码结构（简要）

-   `codes/test_relearn.py`:主要的测试和优化脚本。
-   `options/`: 存放配置文件的目录。
    -   `options/options.py`: 解析配置文件的工具。
-   `utils/`: 包含各种工具函数，如图像处理、日志记录、PSNR/SSIM计算等。
-   `data/`: 数据集加载相关的模块。
-   `models/`: 模型创建和处理相关的模块。

## 待办

-   [ ] 添加更详细的安装指南。
-   [ ] 提供预训练模型和示例配置。
-   [ ] 进一步优化代码结构和可读性。

---

