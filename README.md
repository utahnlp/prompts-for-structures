![Promptly Predicting Structures](https://github.com/utahnlp/prompts-for-structures/blob/dev_basline/logo.png)

This project deals with the idea of using prompts for structured prediction tasks. The idea is to break a structured prediction tasks into independent components which can be queried to a language model using prompts. Following this, inference algorithms are used to enforce structural constraints for the structured prediction task in question.

## Getting Started 
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
3. Create necessary results and dump folders.
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ mkdir -p dumps results
  ```
4. Install package requirements.
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ pip install -r requirements.txt
  ```
5. Install gurobipy. Install the Gurobi Optimzer (https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer-). You'll need a Gurobi licence to use the optimizer. If you are in academia, you can obtain one at: https://www.gurobi.com/academia/academic-program-and-licenses/
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ pip install gurobipy
  ```
6. Clone the official CoNLL coreference scorer library:
  ```console
  (<venv_name>)foo@bar:prompts-for-structures$ git clone https://github.com/conll/reference-coreference-scorers.git
  ```
 
## Running Existing Experiments
1. The required data can be downloaded from this [link](https://drive.google.com/file/d/1-xF1UFClkNdQti0bnKoKPQDyL3OOn1tH/view?usp=sharing). Extract the data in the project directory.
2. You just require the config file to run the experiments for a task/dataset. Existing config files are stored in the `config_files/` directory.
3. Run the zero and few-shot experiments by running the following command:
  ``` console
  (<venv_name>)foo@bar:prompts-for-structures/src$ python model.py --config_file <config_file_path>
  ```
4. Similarly., run the iterative prompting experiments. All the config files for the GPT experiments are located under `config_files/itr_configs/`:
  ``` console
  (<venv_name>)foo@bar:prompts-for-structures/src$ python model_itr.py --config_file <config_file_path>
  ```
5. Run the GPT-4 experiments in a similar fashion. All the config files for the GPT experiments are located under `config_files/gpt_configs/`:
  ``` console
  (<venv_name>)foo@bar:prompts-for-structures/src$ python model_gpt.py --config_file <config_file_path>
  ```
6. SRL metrics and inconsistencies for the unconstrained and the constrained systems will be printed on running the commands mentioned above. However, for coreference experiments only F1 and inconsistencies will be printed. To obtain the CoNLL scores, you must find three files in the `results/` directory with the relevant prefix given in the config file and the suffixes `gold`, `uncons`, and `cons` respectively. These contain the gold clusters, clusters post constrained with R2L inference, and clusters post All-link inference. To compute the CoNLL scores, as an instance, for the All-link constrained clusters you may run:
 ``` console
  (<venv_name>)foo@bar:prompts-for-structures/src$ perl ./reference-coreference-scorers/scorer.pl all <path_to_gold_file> <path_to_cons_file>
  ```


Each config file pertains to a single experiment. In addition, we have added two flags: `read_generated` and `read_inferences`. When `read_generated` is set, it reads the model generation dumps corresponding to the task, dataset, model and read_spec. When `read_inferences` is set, it reads the post-infernce model dumps corresponding to the task, dataset, model and read_spec. When not set, these steps are executed and the data is dumped according to the dump_spec.

### Hardware and Time
Most of the zero-shot experiments with sub-3 billion and equivalent models were run on an NVIDIA TITAN RTX machine (24 GB VRAM). A few 3-billion experiments were also run on an NVIDIA A40 GPU (40 GB VRAM). All 11 billion experiments and fine-tuning experiments were performed on an NVIDIA (80 GB VRAM). Benchmarks on generation and inference times are given in the paper appendix.

7. Head on to the [`src/supervised_models/`](https://github.com/utahnlp/prompts-for-structures/tree/main/src/supervised_models) directory for information about running the supervised model experiments.


---
## Adding your Experiments

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
1. **Preprocessing:** This step processes your data from the data files mentioned in the config file. The preprocessing should result in a DataFrame with each row referring to a component of the structured predicition task. Ad-hoc functions should be imported, and added to the function dictionary under the relevant task_name and dataset_name in the `src\preprocessing.py`.
2. **Prompts:** This step deals with generating prompts from the processed data. The input to this step is a DataFrame and the config details. The output of this step is a parallel list of input prompts and their corresponding gold answers. Ad-hoc functions should be imported, and added to the function dictionary under the relevant task_name and dataset_name in the `src\prompts.py`. If your experiment requires restriction on the output (say, just 'Yes' or 'No'), these can be added for the task name and dataset name in the `restrict_vocab` method in `src\utils.py`. The prompts generated are then fed to the language model to generate candidates for the prompt.
3. **Inference:** This step deals with performing inference over generated responses to enforce structural constraints. It takes the processed data, the language model generations and a sanity check flag as input. The sanity check flag is to equip developers debug their inference protocol by using the gold answers as the top-rated answer. The output of this method should be lsit of model outputs after inference. Ad-hoc functions should be imported, and added to the function dictionary under the relevant task_name and dataset_name in the `src\inference.py`.
4. **Evaluate:** This step deals with performing evaluation for generations- unconstrained and constrained. The input is the processed data, the predicitions which need to be evaluated, and a supplementary dictionary which can be used to pass any other relevant information like saving eval files and so on. Ad-hoc functions should be imported, and added to the function dictionary under the relevant task_name and dataset_name in the `src\evaluate.py`.

Run your code using: 
```console 
(<venv_name>)foo@bar:prompts-for-structures/src$ python model.py --config_file <config_file_path> 
```
In addition, we have added two flags: `read_generated` and `read_inferences`. When `read_generated` is set, it reads the model generation dumps corresponding to the task, dataset, model and read_spec. When `read_inferences` is set, it reads the post-infernce model dumps corresponding to the task, dataset, model and read_spec. When not set, these steps are executed and the data is dumped in accordance to the dump_spec.


---
### Cite Us
To be added soon.
