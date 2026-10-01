#!/bin/bash
# VirtualBox 虚拟机配置脚本

echo "配置VirtualBox虚拟机..."

# 创建虚拟机
VBoxManage createvm --name "ComputerUseTest" --ostype Windows10_64 --register

# 配置虚拟机
VBoxManage modifyvm "ComputerUseTest" --memory 4096 --cpus 2
VBoxManage modifyvm "ComputerUseTest" --vram 128
VBoxManage modifyvm "ComputerUseTest" --graphicscontroller vboxsvga
VBoxManage modifyvm "ComputerUseTest" --clipboard-mode bidirectional
VBoxManage modifyvm "ComputerUseTest" --draganddrop bidirectional

# 创建虚拟硬盘
VBoxManage createmedium disk --filename "ComputerUseTest.vdi" --size 50000

# 添加存储控制器
VBoxManage storagectl "ComputerUseTest" --name "SATA" --add sata --controller IntelAhci
VBoxManage storageattach "ComputerUseTest" --storagectl "SATA" --port 0 --device 0 --type hdd --medium "ComputerUseTest.vdi"

# 网络配置
VBoxManage modifyvm "ComputerUseTest" --nic1 nat
VBoxManage modifyvm "ComputerUseTest" --natpf1 "ssh,tcp,,2222,,22"

# 共享文件夹
VBoxManage sharedfolder add "ComputerUseTest" --name "shared" --hostpath "$(pwd)"

echo "虚拟机配置完成!"
echo "请手动安装Windows 10系统"
echo "安装后运行以下命令启动:"
echo "VBoxManage startvm \"ComputerUseTest\""