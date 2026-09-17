@echo off
echo ========================================
echo Computer Use Model 虚拟机测试脚本
echo ========================================
echo.

echo 1. 检查Docker是否安装...
docker --version
if %errorlevel% neq 0 (
    echo 错误: 请先安装Docker
    echo 下载地址: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

echo 2. 构建Docker镜像...
docker build -t computer-use-model .
if %errorlevel% neq 0 (
    echo 错误: Docker镜像构建失败
    pause
    exit /b 1
)

echo 3. 启动虚拟机测试环境...
echo    - 测试环境将启动Xvfb虚拟显示
echo    - 收集数据并训练模型
echo    - 可通过VNC连接查看: localhost:5900
echo.

docker run -it --rm ^
    -p 5900:5900 ^
    -v %CD%\computer_use_data_advanced:/app/computer_use_data_advanced ^
    -v %CD%\computer_use_model_advanced.pth:/app/computer_use_model_advanced.pth ^
    --shm-size=2g ^
    computer-use-model

echo 4. 测试完成!
pause