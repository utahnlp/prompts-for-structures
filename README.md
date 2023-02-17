# Prompts for Structures
This project deals with the idea of using prompts for structured prediction tasks. The idea is to break a structured prediction tasks into independent components which can be queried to a language model using prompts. Following this, inference algorithms are used to enforce structural constraints for the structured prediction task in question.

## Getting Started 
---
Supported Python Version: 3.8+<br>
To get started with the project, follow the steps mentioned below:
1. Clone the repository to your local working directory.
```console
foo@bar:~$ git clone https://github.com/caffeine96/prompts-for-structures.git
```
2. Enter the project directory. Create a new virtual environment and activate it.
```console
foo@bar:~$ cd prompts-for-structures
foo@bar:~$ python -m venv <venv_name>
foo@bar:~$ source activate <venv_name>/bin/activate
(<venv_name>)foo@bar:~$
```
3. Install package requirements.
```console
(<venv_name>)foo@bar:~$ pip install -r requirements.txt
```
4. Install gurobipy.
```console
(<venv_name>)foo@bar:~$ pip install gurobipy
```
  Install the Gurobi Optimzer (https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer-). You'll need a Gurobi licence to use the optimizer. If you are in academia, you can obtain one at: https://www.gurobi.com/academia/academic-program-and-licenses/
