#!/bin/bash

# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages 
sudo apt-get update -y

# Install OS headers 
sudo apt-get install linux-headers-$(uname -r) -y

# Install git 
sudo apt-get install git -y

# install Neuron Driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Tools 
sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH

# Install Python venv 
sudo apt-get install -y python3.8-venv g++ 

# Create Python venv
python3.8 -m venv aws_neuron_venv_pytorch 

# Activate Python venv 
source aws_neuron_venv_pytorch/bin/activate 
python -m pip install -U pip

# Clone repository
cd /home/ubuntu
su - ubuntu -c "git clone https://github.com/bencemol/aws-neuron-benchmark.git"

# Install dependencies
cd aws-neuron-benchmark
su - ubunty -c "pip3 install -r requirements.txt"
