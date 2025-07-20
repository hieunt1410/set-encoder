# # Uninstall CUDA 12.9
# sudo rm -rf /usr/local/cuda-12.9
# sudo apt-get remove --purge cuda cuda-12-9
# sudo apt-get autoremove
# sudo apt-get remove --purge nvidia-driver-*
# sudo apt-get autoremove

# # Install compatible driver
# sudo apt-get install nvidia-driver-530

# # Install CUDA 12.1
# wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
# sudo sh cuda_12.1.1_530.30.02_linux.run
# echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# source ~/.bashrc


python3.11 -m venv env
source env/bin/activate

pip install -r requirements.txt

sudo apt install -y libsqlite3-dev openjdk-21-jdk
sudo update-java-alternatives --set java-1.21.0-openjdk-amd64
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc