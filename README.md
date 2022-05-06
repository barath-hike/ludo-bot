# LudoRL

## Pre-requisites
The training code has been run using python 3.6. The following dependencies are required:
tensorflow 2.5.0
tensorflow_probability 0.11.1
tqdm

Alternatively an image can be created using the Dockerfile provided 

## Training
To train an algorithm use the train.py script. Within this file is the option to choose which runner to import and how many episodes.
The runner can be selected from the relevant import at the top of the file. Single DQN is already selected for 40,000 training episodes.
Each runner will execute the training for the relevant algorithm. Hyperparameters for each agent can be changed within their respective classes
This will create two folders: results and model_output. Within the model_output file will be the agent weights generated during training and the results file will contain the reward for each player and the episode lengths.
Agents that use a neural network approximator will output an hdf5 file with the network weights. Tabular agents will output a simple .txt file with the tabular function stored within.

Model weights can be loaded into the weights variable in play.py to play against the trained agent. Some existing weights from agents trained within this paper have been included also.

