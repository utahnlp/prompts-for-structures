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
  foo@bar:prompts-for-structures$ python -m venv <venv_name>
  foo@bar:prompts-for-structures$ source activate <venv_name>/bin/activate
  (<venv_name>)foo@bar:~$
  ```
3. Install package requirements.
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ pip install -r requirements.txt
  ```
4. Install gurobipy. Install the Gurobi Optimzer (https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer-). You'll need a Gurobi licence to use the optimizer. If you are in academia, you can obtain one at: https://www.gurobi.com/academia/academic-program-and-licenses/
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ pip install gurobipy
  ```
 
## Running Existing Experiments
---
1. To obtain the required data, please contact the authors of the repo: maitrey.mehta@utah.edu
2. You just require the config file to run the experiments for a task/dataset. Existing config files are stored in the `config_files/` directory.
3. Run the experiment by running the following command:
  ``` console
  (<venv_name>)foo@bar:prompts-for-structures/src$ python model.py --config_file <config_file_path>
  ```
