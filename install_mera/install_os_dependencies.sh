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

total_ram=$(free --giga | awk 'NR==2{print $2}')
echo "-- total RAM is ${total_ram}GB --"
echo "For model compilation, at least 32GB is recommended."
echo "For larger LLMs compilation, at least 64GB is recommended."

# wait for message to be read
echo ".. installing dependencies .."
sleep 4s

# Install system wide dependencies
sudo apt update && sudo apt install -y gcc-12 llvm-14 libgomp1 ocl-icd-libopencl1 software-properties-common \
    libgoogle-glog0v5 libboost-graph-dev virtualenv wget build-essential libssl-dev python3-dev python3-venv git

# upgrade libstdc++ version
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install -y libstdc++6
