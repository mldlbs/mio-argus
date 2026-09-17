# Computer Use Model 虚拟机测试指南

## 方案一：Docker测试（推荐）

### 前提条件
- 安装Docker Desktop: https://docs.docker.com/desktop/install/windows-install/

### 步骤
1. 构建Docker镜像：
   ```bash
   docker build -t computer-use-model .
   ```

2. 运行测试：
   ```bash
   docker run -it --rm -p 5900:5900 computer-use-model
   ```

3. 查看测试过程：
   - 使用VNC客户端连接 `localhost:5900`
   - 或使用浏览器访问 `http://localhost:5900`

## 方案二：VirtualBox虚拟机

### 前提条件
- 安装VirtualBox: https://www.virtualbox.org/

### 步骤
1. 运行配置脚本：
   ```bash
   bash setup_virtualbox.sh
   ```

2. 手动安装Windows 10系统

3. 在虚拟机中安装Python和依赖

4. 运行测试脚本

## 方案三：VMware虚拟机

### 前提条件
- 安装VMware Workstation Player

### 步骤
1. 创建Windows 10虚拟机
2. 配置共享文件夹
3. 在虚拟机中运行测试

## 测试内容

### 数据收集测试
```bash
python collect_data_advanced.py
```

### 模型训练测试
```bash
python train_computer_use_advanced.py
```

### 模型测试
```bash
python test_advanced_model.py
```

## 注意事项

1. **分辨率设置**：确保虚拟机分辨率为1920x1080
2. **网络连接**：确保能访问互联网下载依赖
3. **性能要求**：建议至少4GB内存，2个CPU核心
4. **安全隔离**：虚拟机环境可防止误操作影响主机

## 监控和调试

### 查看Docker日志
```bash
docker logs computer-use-model-test
```

### 进入Docker容器
```bash
docker exec -it computer-use-model-test bash
```

### 查看VNC连接
- 下载VNC客户端: https://www.realvnc.com/en/connect/download/viewer/
- 连接地址: `localhost:5900`

## 故障排除

### 问题1: Docker构建失败
- 检查Docker是否正常运行
- 确保有足够的磁盘空间

### 问题2: 虚拟机性能差
- 增加虚拟机内存和CPU核心数
- 关闭不必要的后台程序

### 问题3: pyautogui无法工作
- 确保虚拟机有图形界面
- 检查DISPLAY环境变量设置