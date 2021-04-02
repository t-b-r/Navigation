# Project 1: Navigation

### Overview


In this project, we implement a deep q network to train an agent to learn how to navigate OpenAI Gym's Banana environment. In this episodic environment, the objective is to collect as many yellow bananas as possible while avoiding blue bananas. The agent is capable of moving forwards, backwards, left, or right, and perceives the environment via a 37-dimensional ray-based vector around the agent's forward direction. Each episode terminates after 1,000 time steps have been traversed, and we consider the agent to have learned the environment when it is able to achieve a score of +13.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.


### Instructions

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

3. Add the files dqn_agent.py, dqn_model.py, and Navigation.ipynb to the directory p1_navigation, or alternatively, clone the repo into that folder.

4. Run all cells in Navigation.ipynb

### The files

- **Navigation.ipynb**: Contains the code to run the agent in the environment
- **dqn_agent.py**: Code for the agent
- **dqn_model.py**: Code defining the deep Q Network 
- **checkpoint.pth**: Pytorch model weights