# RTX 5070Ti显卡CUDA兼容性解决方案

## 📋 问题概述

RTX 5070Ti等新一代显卡基于Ada Lovelace架构，需要较新的CUDA版本和驱动支持。旧版本的PyTorch可能无法正确识别和使用这些新显卡，导致以下问题：

- CUDA相关错误
- PyTorch无法识别GPU
- 模型被强制加载到CPU运行
- 性能大幅下降

---

## 🔍 环境检查

### 第一步：检查当前状态

```powershell
# 1. 检查显卡型号和驱动版本
nvidia-smi

# 2. 检查当前PyTorch版本和CUDA支持
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'PyTorch CUDA版本: {torch.version.cuda}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU检测到"}')"

# 3. 检查CUDA工具包版本（如果已安装）
nvcc --version
```

### 第二步：确认问题类型

**如果出现以下情况，说明需要更新**：
- 驱动版本 < 560.x
- PyTorch CUDA版本 < 12.0
- `torch.cuda.is_available()` 返回 `False`
- GPU名称显示为"无GPU检测到"

---

## 🛠️ 解决方案

### 方案一：更新驱动程序（必需）

1. **下载最新NVIDIA驱动**
   - 访问：https://www.nvidia.com/drivers
   - 选择：RTX 5070Ti
   - 下载版本：560.x 或更高

2. **安装驱动**
   ```powershell
   # 建议先卸载旧驱动，然后重启安装新驱动
   # 安装完成后重启计算机
   ```

3. **验证驱动安装**
   ```powershell
   nvidia-smi
   # 应显示RTX 5070Ti和最新驱动版本
   ```

### 方案二：更新PyTorch（推荐）

1. **激活虚拟环境**
   ```powershell
   # 进入VibeVoice项目目录
   cd e:\User\Documents\GitHub\VibeVoice
   
   # 激活虚拟环境
   vibevoice-env\Scripts\Activate.ps1
   ```

2. **卸载旧版本PyTorch**
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   ```

3. **安装支持CUDA 12.1的PyTorch**
   ```powershell
   # 安装最新版本（支持RTX 5070Ti）
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **验证安装**
   ```powershell
   python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'PyTorch CUDA版本: {torch.version.cuda}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU检测到"}')"
   ```

### 方案三：使用Conda管理（备选方案）

如果pip方式遇到问题，可以使用conda：

1. **安装Miniconda**
   - 下载：https://docs.conda.io/en/latest/miniconda.html

2. **创建新环境**
   ```powershell
   conda create -n vibevoice-cuda python=3.11
   conda activate vibevoice-cuda
   ```

3. **安装PyTorch**
   ```powershell
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. **安装VibeVoice依赖**
   ```powershell
   cd e:\User\Documents\GitHub\VibeVoice
   pip install -e .
   ```

---

## ✅ 验证解决方案

### 完整验证脚本

```powershell
# 运行完整验证
python -c "
import torch
print('='*50)
print('RTX 5070Ti CUDA兼容性检查')
print('='*50)
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'PyTorch CUDA版本: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    # 简单的GPU计算测试
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    print('GPU计算测试: 通过')
else:
    print('警告: GPU不可用，请检查驱动和CUDA安装')
print('='*50)
"
```

### 预期输出

```
==================================================
RTX 5070Ti CUDA兼容性检查
==================================================
PyTorch版本: 2.4.0+cu121
CUDA可用: True
PyTorch CUDA版本: 12.1
GPU数量: 1
GPU名称: NVIDIA GeForce RTX 5070 Ti
GPU显存: 16.0 GB
GPU计算测试: 通过
==================================================
```

---

## 🚀 重新启动VibeVoice

验证成功后，重新启动VibeVoice服务：

```powershell
# 确保在正确的环境中
vibevoice-env\Scripts\Activate.ps1

# 启动服务（包含FFmpeg路径）
$env:PATH += ";C:\ffmpeg\bin"; python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B
```

---

## 🔧 故障排除

### 问题1：驱动安装失败
**解决方案**：
- 使用DDU工具完全卸载旧驱动
- 以管理员身份安装新驱动
- 确保Windows更新已完成

### 问题2：PyTorch安装失败
**解决方案**：
```powershell
# 清理pip缓存
pip cache purge

# 使用清华镜像源
pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ --extra-index-url https://download.pytorch.org/whl/cu121
```

### 问题3：CUDA版本冲突
**解决方案**：
- 卸载所有CUDA工具包
- 仅依赖PyTorch内置的CUDA运行时
- 避免系统级CUDA安装

### 问题4：显存不足
**解决方案**：
```powershell
# 使用较小的模型
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B

# 或者设置显存限制
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## ⚠️ 重要提醒

1. **备份环境**：在更新前备份当前工作环境
2. **网络要求**：下载过程需要稳定的网络连接
3. **重启要求**：驱动更新后必须重启计算机
4. **版本兼容**：确保PyTorch版本 >= 2.1.0
5. **环境隔离**：建议在虚拟环境中进行所有操作

---

## 📞 技术支持

如果按照本方案仍无法解决问题，请提供以下信息：

1. `nvidia-smi` 输出
2. PyTorch版本检查结果
3. 具体错误信息
4. 操作系统版本
5. 之前的CUDA/PyTorch安装历史

---

**更新日期**: 2025年1月
**适用显卡**: RTX 5070Ti, RTX 5080, RTX 5090等新一代显卡
**测试环境**: Windows 10/11, Python 3.8+