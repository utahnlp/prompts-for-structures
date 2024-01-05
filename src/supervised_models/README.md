## Supervised Experiments

To run the supervised model experiments follow these steps:

1. Create necessary models and dumps folders.
   ```console
   (<venv_name>)foo@bar:prompts-for-structures/src/supervised_models$ mkdir -p dumps ./../../models
   ```
2. Install package requirements.
   ```console
   (<venv_name>)foo@bar:prompts-for-structures/src/supervised_models$ pip install -r requirements.txt
   ```
3. To train the model for an SRL dataset ("wiki" or "qasrl2") run:
   ```console
   (<venv_name>)foo@bar:prompts-for-structures/src/supervised_models$ python run_srl_t5_model.py --dataset_name <dataset_name>
   ```
4. To evaluate the model for an SRL dataset ("wiki" or "qasrl2") run:
   ```console
   (<venv_name>)foo@bar:prompts-for-structures/src/supervised_models$ python run_srl_t5_model.py --dataset_name <dataset_name> --mode test --best_model <best_model_epoch>
    ```
5. To train the model for a coreference dataset ("ecbp", "ontonotes", or "genia") run:
   ```console
   (<venv_name>)foo@bar:prompts-for-structures/src/supervised_models$ python run_coref_macaw_model.py --dataset_name <dataset_name>
   ```
6. To evaluate the model for a coreference dataset ("ecbp", "ontonotes", or "genia") run:
   ```console
   (<venv_name>)foo@bar:prompts-for-structures/src/supervised_models$ python run_srl_macaw_model.py --dataset_name <dataset_name> --mode test --best_model <best_model_epoch>
   ```
