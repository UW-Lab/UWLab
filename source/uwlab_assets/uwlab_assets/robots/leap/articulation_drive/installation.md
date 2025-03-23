# Leap Hand Hardware Installation Notes

## Installation Steps

### 1. Install Required Packages
To control the hand, install the necessary Python package:
```bash
pip install dynamixel-sdk
```

### 2. Install Additional Packages for Teleoperation with Rokoko
If you need to teleoperate the hand with Rokoko, install these additional packages
```bash
pip install lz4 pyrealsense2==2.53.1.4623
sudo apt install librealsense2=2.53.1-0~realsense0.8251
sudo apt install librealsense2-gl=2.53.1-0~realsense0.8251
sudo apt install librealsense2-net=2.53.1-0~realsense0.8251
sudo apt install librealsense2-utils=2.53.1-0~realsense0.8251
```
and configure the following firewall settings:
```bash
sudo ufw allow from your_mac_ip/24
```
