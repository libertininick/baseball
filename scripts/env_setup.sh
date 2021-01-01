#!/bin/bash
conda create -y -n baseball_env python=3.8 pip
source activate baseball_env
python -m pip install -r requirements.txt
python -m ipykernel install --user --name baseball_env
