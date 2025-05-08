#!/bin/bash

# check os and kernel version
ubuntu_ver="$(lsb_release -sr)"
echo "-- ubuntu version is $ubuntu_ver. --"

# Check if version is not 20.04 or 22.04
if [[ "$ubuntu_ver" != "20.04" ]] &&
   [[ "$ubuntu_ver" != "22.04" ]]; then
    echo "ERROR: ubuntu version needs to be 20.04 or 22.04."
    exit 1
fi

# Install system wide dependencies
sudo apt update && sudo apt install -y gcc-12 llvm-14 libgomp1 ocl-icd-libopencl1 software-properties-common \
    libgoogle-glog0v5 libboost-graph-dev virtualenv wget build-essential libssl-dev python3-dev python3-venv git

# upgrade libstdc++ version
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install -y libstdc++6

# FTDI library (for power measurement comm through usb)
# mkdir libft
# cd libft
# wget https://ftdichip.com/wp-content/uploads/2022/06/libft4222-linux-1.4.4.170.tgz
# tar xvzf libft4222-linux-1.4.4.170.tgz
# sudo ./install4222.sh
# cd -
# rm -rf libft
# sudo ldconfig -v
