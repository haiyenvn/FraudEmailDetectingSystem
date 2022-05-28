# update instance
sudo yum update -y

# install python 3.8
sudo yum install -y amazon-linux-extras
sudo amazon-linux-extras enable python3.8
sudo yum install python38
sudo yum install git

# clone project and install required libraries
git clone https://github.com/haiyenvn/FraudEmailDetectingSystem.git
cd FraudEmailDetectingSystem
sudo python3.8 -m pip install -r requirements.txt

# run web app
streamlit run model_deployment.py