# Aspect-Based-Sentiment-Analysis
Implementation of models for the *aspect term identification* and *aspect term polarity classification* tasks.

## Description

An explanation regarding task and tested models can be found in the *report.pdf* file.

## OS

I developed this code on an Ubuntu 20.04.2 LTS machine.

## How to run

### Requirements

* [conda](https://docs.conda.io/projects/conda/en/latest/index.html);

* [docker](https://www.docker.com/), to avoid any issue pertaining code runnability.

### Notes

Unless otherwise stated, all commands here are expected to be run from the root directory of this project.

### Setup Environment

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

*test.sh* essentially setups a server exposing the model through a REST Api and then queries this server, evaluating the model. So first, you need to install Docker:

```
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding.

The model will be exposed through a REST server. In order to call it, we need a client. The client has been written
in the evaluation script, but it needs some dependencies to run. We will be using conda to create the environment for this client.

```
conda create -n nlp2021-hw2 python=3.7
conda activate nlp2021-hw2
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```
conda activate nlp2021-hw2
bash test.sh data/restaurants_dev.json
```

The file *data_dev.json* is just the union of the other two dev files.

## Additional instructions

In order to be able to test the models you should download them here:

- [model for *aspect term identification* (model A)](https://drive.google.com/file/d/1cq-WaoPW4aG8t_a0C99gIMk2DTKwru23/view?usp=sharing);
- [model for *aspect term polarity classification* (model B)](https://drive.google.com/file/d/1QfKNRBGN3a8VHr8HMLwNSsAYl4F99P4H/view?usp=sharing).

They have to be placed in the *model/* folder.

