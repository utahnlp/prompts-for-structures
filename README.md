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
  (<venv_name>)foo@bar:prompts-for-structures$
  ```
3. Create necessary data and dump folders.
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ mkdir -p data, dumps
  ```
4. Install package requirements.
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ pip install -r requirements.txt
  ```
5. Install gurobipy. Install the Gurobi Optimzer (https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer-). You'll need a Gurobi licence to use the optimizer. If you are in academia, you can obtain one at: https://www.gurobi.com/academia/academic-program-and-licenses/
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

## Adding your Experiments
---
The codebase follows a centralized generation and inference pipeline. There are two facets to adding your task/dataset to the codebase: a) config file, and b) Ad-hoc code.
### Config File
The config file is the only input the system. The parameters in the file fixes the experimental setting. These parameters are explained briefly here:
- Meta<br>
  a. `task_name`: The parameter sets the structured prediction task which is undertaken. All datasets for this task must use the same value for the   task_name. If this is the first instance of the task, feel free to select a name.<br>
  b. `dataset_name`: The dataset name used for the experiment. Use names consistently across runs.
- Data<br>
  a. `data_dir`: The directory where the data resides. Recommend fixing this to `./../data/` unless need be otherwise.<br>
  b. `data_file`: The path with respect to data_dir to your data file(s)
- Run<br>
  a. `mode`: Mode of the run. This will remain `predict` unless model training/fine-tuning is required.
- Model<br>
  a. `model`: The model name to be used as your language model. Currently supported- `t5`(t5-large), `t5-3b`, `t5-11b`, `macaw-3b`, and `unified-qa`.
- Prompt <br>
  a. `prompt_type`: Will remain `discrete` for now. Functionalities for adding in-context examples and continous prompts coming soon.
  b. `prompt_style`: A flexible parameter to tune how you want to position your task to the model. For example, as a QA task, Multiple-choice, etc.
  c. `context_style`: A flexible parameter to tweak contexts for various experiments.
- Dumps <br>
  a. `dump_spec`: An infix to add when gold data and generations are dumped on disk. The dump will already contain information about the task, dataset and the model used. Any other spec can be added in this string.
  b. `read_spec`: Infix which should be used to read any existing dumps

### Adding Your Code
The config file is the first step. This file is passed while running `src\model.py` file which is where the execution starts. The execution is broken into multiple steps. Each step has a nodal file under the `src\` directory. Each nodal file references the ad-hoc methods utilized for certain tasks and datasets. All ad-hoc files are stored under `src\tasks\<task_name>\<dataset_name>\`. Ideally, each steop should have its own file for readibility. The role of the nodal file is to ensure that each step has a consistent inputs and outputs, hence promoting re-usability. The steps in the execution are as follows:
1. **Preprocessing:** The processes your data from the data files mentioned in the config file. The preprocessing should result in a DataFrame with each row referring to a component of the structured predicition task. Ad-hoc functions should be imported, and added to the function dictionary under the relevant task_name and dataset_name in the `src\preprocessing.py`.
2. 
