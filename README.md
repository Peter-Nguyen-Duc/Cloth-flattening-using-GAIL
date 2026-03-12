# Generative Adversarial Imitation Learning for flattening

This repository contains a framework for using GAIL in MuJoCo for cloth flattening.


## Guide:

### setup virtual environment (optional)
$ python -m venv venv
$ source venv/bin/activate


### install requirements
$ pip install -r requirements.txt


### Run GAIL with PPO 
$ python -m learning.AIRL.main



## Repository contains:
- PPO implementation
- GAIL implementation (Not sure if works anymore)

Some experimentation was also done with:
- AIRL (cannot guarantee that it works)



This repository also consists of expert demonstrations and MuJoCo environments for:



# Code sources
- Mujoco environment design
https://gitlab.sdu.dk/sdurobotics/teaching/mj_sim


- PPO inspired implementation 
https://github.com/reinforcement-learning-kr/lets-do-irl


- GAIL implementation[
https://github.com/reinforcement-learning-kr/lets-do-irl



