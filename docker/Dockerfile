#Requirements
# - install docker for your host operating system https://docs.docker.com/get-docker/ 
# - git clone or download mal2 source from https://mal2git.x-t.at/root/mal2/tree/dashboard 
# - get fake-shop db access token (see mal2/eCommerce/resources/fakeshopdb_api_access_token_readme.txt)
# - extended user documentation (see mal2/eCommerce/docs/Installationsanleitung_und_UserGuide_Dashboard.pdf)

#Docker Build
#cd to mal2-model/eCommerce and run 'docker build -t ecommerce_mal2-fake-shop-dashboard -f docker/Dockerfile .'

#Run1: With VNC graphical UI
#'docker run --name mal2-dashboard-vnc  --mount type=bind,source=/media/sf_win_repositories/mal2/mal2/eCommerce,target=/root/mal2 -p 6080:80 -p 5900:5900 ecommerce_mal2-fake-shop-dashboard'
#Access the Application
#launch http://127.0.0.1:6080/#/ or connect on port 5900 using a VNC Client

#Run2: Shell only
#'docker run --name mal2-dashboard-shell -it --entrypoint /root/Desktop/open_mal2_shell.sh --mount type=bind,source=/media/sf_win_repositories/mal2/mal2/eCommerce,target=/root/mal2  ecommerce_mal2-fake-shop-dashboard'

#Start/Stop 
# to start: docker start mal2-dashboard-vnc - to stop: docker stop mal2-dashboard-vnc
# #Note: if you add --rm in docker run, then the container gets deleted if you exit it 

#-----------------------------------------------------------------------------
#Docker Build Instructions

FROM dorowu/ubuntu-desktop-lxde-vnc:bionic

# for connectivity testing
#RUN apt-get update && apt-get install -y telnet

# set a directory for the app
WORKDIR $HOME/temp/

# copy all the required files for building the container
# other data expected to be mounted at runtime: docker run -v /host/directory:/container/directory
COPY ./requirements.txt /root/Desktop/requirements.txt
COPY ./docker/open_mal2_shell.sh /root/Desktop/open_mal2_shell.sh
COPY ./docker/kosoh_logo_classifier-0.1-py3-none-any.whl /root/Desktop/kosoh_logo_classifier-0.1-py3-none-any.whl
RUN sudo chmod +x /root/Desktop/open_mal2_shell.sh

# add package containing pip and install pip + wget
# NOTE: gcc and build-essential are required for shap compilation reasons!
RUN sudo apt update && sudo apt install wget && sudo apt-get install gcc build-essential -y

# install anaconda
#RUN sudo mkdir -p $HOME/miniconda
RUN sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && sudo bash miniconda.sh -b -p $HOME/miniconda

# add conda to the path
#ENV causes issues with the vnc server due to python conflicts
#ENV PATH=$HOME/miniconda/bin:$PATH
RUN sudo $HOME/miniconda/bin/conda init bash

# create a new python 3.7.6 environment mal2-model and install pip requirements
# see issues with conda activate https://pythonspeed.com/articles/activate-conda-dockerfile/ 
RUN sudo /root/Desktop/open_mal2_shell.sh

# install chrome and chromedriver
# We need wget to set up the PPA and xvfb to have a virtual screen and unzip to install the Chromedriver
RUN apt-get install -y xvfb unzip

# Set up the Chrome PPA
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list

# Update the package list and install chrome
RUN apt-get update -y
RUN apt-get install -y google-chrome-stable

# Set up Chromedriver Environment variables
ENV CHROMEDRIVER_VERSION 81.0.4044.69
ENV CHROMEDRIVER_DIR /chromedriver
RUN mkdir $CHROMEDRIVER_DIR

# Download and install Chromedriver
RUN wget -q --continue -P $CHROMEDRIVER_DIR "http://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
RUN unzip $CHROMEDRIVER_DIR/chromedriver* -d $CHROMEDRIVER_DIR

# Put Chromedriver into the PATH
ENV PATH $CHROMEDRIVER_DIR:$PATH

# define the port number the container should publish (=external access)
# allows you to acces VNC at http://127.0.0.1:6080/#/ -p 6080:80 -p 5900:5900
