#/bin/sh

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb

sudo apt-get update
sudo apt-get install cuda
sudo apt-get install nvidia-gds
sudo reboot 0
