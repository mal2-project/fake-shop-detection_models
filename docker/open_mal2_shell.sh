#!/bin/bash

echo "hello MAL2 dashboard"
ENVS=$(/home/ubuntu/miniconda/bin/conda env list | awk '{print $1}' )
echo $ENVS

eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"

if [[ $ENVS == *"mal2-model"* ]]; then
	echo "activating conda env mal2-model"
	conda activate mal2-model
	python --version
	
	#run a shell
	bash -c "cd /root/mal2; /home/ubuntu/miniconda/bin/conda init bash; conda env list; echo 'type: conda activate mal2-model'; exec bash"

else 
   echo "create conda env mal2-model"
   conda create --name mal2-model python=3.7.6 -y
   conda activate mal2-model
   pip install -r /root/Desktop/kosoh_logo_classifier-0.1-py3-none-any.whl
   pip install -r /root/Desktop/requirements.txt
   rm /root/Desktop/requirements.txt
   python --version
fi;

#exit