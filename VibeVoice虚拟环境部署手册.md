# VibeVoice 虚拟环境部署手册

## 📋 部署概览

本手册将指导您使用Python虚拟环境安全部署VibeVoice项目，避免影响系统全局环境。

### 🎯 部署目标
- 创建隔离的Python虚拟环境
- 安装VibeVoice及其依赖
- 启动Gradio演示界面
- 验证核心功能

### ⚙️ 系统要求
- Windows 10/11
- Python 3.8+
- **FFmpeg** (必需，用于音频处理)
- NVIDIA GPU（推荐，支持CUDA）
  - **RTX 5070Ti等新显卡**: 需要CUDA 12.0+和最新驱动
  - **RTX 30/40系列**: CUDA 11.8+即可
  - **GTX系列**: 可能性能较低，建议RTX系列
- 至少8GB可用内存（GPU显存建议8GB+）
- 至少10GB可用磁盘空间

### 📋 预安装依赖
**在开始部署前，请确保已安装以下软件**:

1. **FFmpeg安装**:
   - 下载地址: https://ffmpeg.org/download.html
   - 推荐安装路径: `C:\ffmpeg\bin`
   - 验证安装: 在PowerShell中运行 `ffmpeg -version`

2. **Git** (如需克隆代码):
   - 下载地址: https://git-scm.com/download/win

---

## 🔍 第一步：环境检查

### 1.1 检查Python版本
```bash
python --version
```
**预期输出**: Python 3.8.x 或更高版本

### 1.2 检查FFmpeg（必需）
```bash
ffmpeg -version
```
**预期输出**: 显示FFmpeg版本信息
**如果失败**: 请先安装FFmpeg到 `C:\ffmpeg\bin` 并添加到系统PATH

### 1.3 检查GPU支持（可选但推荐）
```bash
nvidia-smi
```
**预期输出**: 显示GPU信息和驱动版本

### 1.4 检查PyTorch CUDA支持
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```
**预期输出**: 显示PyTorch版本和CUDA可用性

### 1.5 检查显卡兼容性（RTX 5070Ti等新显卡用户必读）
```bash
# 检查显卡型号和驱动版本
nvidia-smi

# 检查CUDA版本兼容性
python -c "import torch; print(f'PyTorch CUDA版本: {torch.version.cuda}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU检测到"}')"
```
**RTX 5070Ti用户特别检查**:
- 驱动版本应 >= 560.x
- PyTorch CUDA版本应 >= 12.0
- 如果CUDA不可用，请参考故障排除Q8

---

## 🏗️ 第二步：创建虚拟环境

### 2.1 创建虚拟环境
```bash
# 在项目根目录创建虚拟环境
python -m venv vibevoice-env
```

### 2.2 激活虚拟环境
```bash
# Windows PowerShell
vibevoice-env\Scripts\Activate.ps1

# Windows CMD
vibevoice-env\Scripts\activate.bat
```

### 2.3 验证虚拟环境激活
```bash
# 检查Python路径（应指向虚拟环境）
where python

# 检查pip路径
where pip
```
**预期输出**: 路径应包含 `vibevoice-env`

---

## 📦 第三步：安装依赖

### 3.1 升级pip
```bash
python -m pip install --upgrade pip
```

### 3.2 安装VibeVoice项目
```bash
# 以开发模式安装项目
pip install -e .
```

### 3.3 验证安装
```bash
# 测试导入
python -c "import vibevoice; print('VibeVoice安装成功！')"

# 检查已安装的包
pip list | findstr vibevoice
```

---

## 🚀 第四步：启动服务

### 4.1 配置FFmpeg（重要）
**⚠️ 必须步骤**: 在启动服务前，需要确保FFmpeg可用

```bash
# 方法1：设置环境变量（推荐）
$env:PATH += ";C:\ffmpeg\bin"

# 方法2：或者在启动命令中指定
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --ffmpeg_path "C:\ffmpeg\bin\ffmpeg.exe"
```

**FFmpeg安装验证**:
```bash
ffmpeg -version
```
**预期输出**: 显示FFmpeg版本信息

### 4.2 启动Gradio演示（1.5B模型）
```bash
# 推荐方式：包含FFmpeg路径
$env:PATH += ";C:\ffmpeg\bin"; python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share
```

### 4.3 启动Gradio演示（7B模型，更稳定）
```bash
# 推荐方式：包含FFmpeg路径
$env:PATH += ";C:\ffmpeg\bin"; python demo/gradio_demo.py --model_path WestZhang/VibeVoice-Large-pt --share
```

### 4.4 访问服务
- **本地访问**: http://localhost:7860
- **公网访问**: 查看终端输出的gradio.live链接

### 4.5 验证启动成功
**正常启动日志应包含**:
```
✓ 模型加载完成
✓ 在demo/voices目录下找到X个音色文件
✓ 加载X个示例脚本
✓ Running on local URL: http://127.0.0.1:7860
```

---

## ✅ 第五步：功能验证

### 5.1 单说话人测试
1. 在Gradio界面选择 "Number of Speakers" = 1
2. 选择说话人声音（如 "Alice"）
3. 输入测试文本："Hello, this is a test of VibeVoice."
4. 点击 "Generate Podcast" 按钮
5. 验证音频生成和播放

### 5.2 多说话人对话测试
1. 选择 "Number of Speakers" = 2
2. 设置 Speaker 1: "Alice", Speaker 2: "Frank"
3. 输入对话脚本：
```
Alice: Welcome to our podcast!
Frank: Thank you for having me, Alice.
Alice: Let's discuss AI technology.
Frank: That's a fascinating topic.
```
4. 生成并验证多说话人效果

### 5.3 文件批处理测试
```bash
# 单说话人文件处理
python demo/inference_from_file.py --model_path microsoft/VibeVoice-1.5B --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

# 多说话人文件处理
python demo/inference_from_file.py --model_path microsoft/VibeVoice-1.5B --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank
```

---

## 🛠️ 故障排除

### 常见问题

**Q1: 虚拟环境激活失败**
```bash
# 如果PowerShell执行策略限制，运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Q2: 模型下载缓慢**
```bash
# 设置Hugging Face镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com
```

**Q3: CUDA内存不足**
- 关闭其他GPU占用程序
- 使用较小的模型（1.5B而非7B）
- 减少batch size或序列长度

**Q4: 中文语音不稳定**
- 使用英文标点符号
- 优先选择7B模型
- 避免特殊字符

**Q5: FFmpeg相关错误**
```
错误信息: "ffmpeg not found" 或音频处理失败
```
**解决方案**:
1. 确认FFmpeg已正确安装到 `C:\ffmpeg\bin`
2. 验证FFmpeg可执行: `ffmpeg -version`
3. 在启动命令前添加路径: `$env:PATH += ";C:\ffmpeg\bin"`
4. 重启PowerShell终端后重新尝试

**Q6: 自定义音色问题**
```
现象: 自定义音色文件能被识别但无法正常生成音频
```
**问题分析**:
- VibeVoice当前版本不支持真正的自定义音色训练
- 仅将音色文件放入 `demo/voices/` 目录无法实现声音克隆
- 系统会识别文件但无法使用其声学特征

**解决方案**:
1. **使用内置音色**: 推荐使用项目提供的10个内置音色
   - Alice, Bob, Charlie, Diana, Eric, Fiona, George, Hannah, Ian, Julia
2. **等待官方更新**: 关注项目更新，训练代码可能在未来版本发布
3. **替代方案**: 考虑使用其他支持声音克隆的开源TTS项目
   - Coqui TTS
   - Real-Time-Voice-Cloning
   - SpeechT5

**Q7: 控制台HLS流媒体错误**
```
错误信息: "net::ERR_ABORTED", "HLS error: levelEmptyError"
```
**问题分析**:
- 这些是音频流播放的暂时性网络错误
- 通常由音频生成速度慢或缓冲问题导致
- 不影响最终音频生成结果

**解决方案**:
1. 忽略这些错误，等待音频生成完成
2. 使用更强的GPU加速音频生成
3. 减少输入文本长度
4. 刷新浏览器页面重新尝试

**Q8: RTX 5070Ti等新一代显卡CUDA兼容性问题**
```
错误信息: CUDA相关错误、PyTorch无法识别GPU、模型加载到CPU
```
**问题分析**:
- RTX 5070Ti基于Ada Lovelace架构，需要较新的CUDA版本支持
- 旧版本的PyTorch可能不支持新一代显卡
- 驱动版本过低会导致CUDA功能异常

**解决方案**:
1. **更新显卡驱动**:
   ```bash
   # 下载最新的NVIDIA驱动（建议560.x或更高版本）
   # 官网: https://www.nvidia.com/drivers
   ```

2. **安装支持新显卡的PyTorch版本**:
   ```bash
   # 卸载旧版本PyTorch
   pip uninstall torch torchvision torchaudio
   
   # 安装最新版本（支持CUDA 12.x）
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **验证CUDA支持**:
   ```bash
   python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU"}')"
   ```

4. **检查CUDA版本兼容性**:
   ```bash
   nvidia-smi  # 查看支持的CUDA版本
   python -c "import torch; print(f'PyTorch CUDA版本: {torch.version.cuda}')"
   ```

**特别注意**:
- RTX 5070Ti需要CUDA 12.0或更高版本
- 确保PyTorch版本 >= 2.1.0
- 如果仍有问题，可尝试使用conda环境管理CUDA依赖

---

## 🔄 环境管理

### 退出虚拟环境
```bash
deactivate
```

### 重新激活虚拟环境
```bash
vibevoice-env\Scripts\Activate.ps1
```

### 删除虚拟环境（如需重新安装）
```bash
# 先退出虚拟环境
deactivate

# 删除环境目录
Remove-Item -Recurse -Force vibevoice-env
```

---

## 📚 参考资源

- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [项目主页](https://microsoft.github.io/VibeVoice)
- [Hugging Face模型](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
- [技术论文](https://arxiv.org/pdf/2508.19205)

---

## ⚠️ 注意事项

1. **模型下载**: 首次运行会自动下载模型文件（约3-15GB）
2. **网络要求**: 需要稳定的网络连接下载模型
3. **硬件要求**: GPU推荐但非必需，CPU也可运行（速度较慢）
4. **新显卡兼容性**: RTX 5070Ti等新一代显卡需要最新的CUDA 12.0+和驱动560.x+
5. **FFmpeg依赖**: 必须正确配置FFmpeg才能正常运行，否则会出现音频处理错误
6. **自定义音色限制**: 当前版本不支持真正的声音克隆，仅支持内置的10个音色
7. **使用限制**: 仅用于研究和开发，不建议商业使用
8. **内容责任**: 生成的音频内容需要用户自行负责
9. **流媒体错误**: 浏览器控制台可能出现HLS相关错误，这是正常现象，不影响音频生成
10. **环境隔离**: 强烈建议使用虚拟环境，避免与系统Python环境冲突

---

**部署完成后，您将拥有一个完全隔离的VibeVoice环境，可以安全地进行语音合成实验！**