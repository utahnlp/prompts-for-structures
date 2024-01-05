## Using the Config Files

In the this project base, config files are used to set different parameters for different experiments. We have provided a few config files for reference. The entire set of experiments in the paper can be performed using these config files, or making minor tweaks (discussed below). Here is a quick summary:

1. The config files for the zero-shot experiments are present under the `zero-shot_configs/` directory. Each file pertains to the T5-3B (for SRL) and Macaw-3B (for coref) experiments for each dataset. To reproduce the model size experiments, you may create new config files by replacing the `model` parameter value with the different model names one by one. The models currectly supported are:<br>
    a. **SRL:** t5-small, t5-base, t5 (for t5-large), t5-3b, t5-11b, flan-t5-xl, unified-qa, unifiedqa-v2 <br>
    b. **Coref:** t5 (for t5-large), t5-3b, t5-11b, macaw-large, macaw-3b, macaw-11b, flan-t5-xl<br>
    The iterative and GPT-4 experiments follow a different script and config file discussed later in this document.

2. For context-style experiments (reported in the appendix), one may set the `context_style` parameter from None (equivalent to Rel. setting), 'highlight' (Hlght), 'full_context' (Full), or "highlight_full_context" (Full + Hlght). Note that context-style experiments are only applicable to Coreference datasets.

3. The config files for the few-shot experiments are present under the `few-shot_configs/` directory. Each file pertains to the T5-3B (for SRL) and Macaw-3B (for coref) experiments for each dataset for three shots and random seed 42. You may change the number of shots by setting the `num_shots` parameter (in the paper 1, 3 and 5 shots are used). You may change the random seed (42,20, and 1984 in the paper) by setting the `shot_seed` parameter. This ensures that a different set of shots in used for that particular run. Additionally, `shot_type` can take a value other than 'same' which ensures a different set of shots is picked for each instance (not used for the paper). The `order_num` parameter changes the order in which the shots are presented. Naturally, it may take any value from 1 to n! where n is the number of shots.

4. The config files for the iterative prompting experiments are present under `itr_configs`. Note that iterative prompting is only performed for the SRL datasets.

5. The config files for GPT-4 experiments are present under `gpt_configs`. Note that GPT-4 experiments are only performed for the SRL datasets.  


**IMPORTANT:** To avoid overwriting, for every experimental setting where anything other than the model and dataset are changed, ensure that the `dump_spec` and `read_spec` are changed to something unique. The `dump_spec` provides a file name infix which is used to create the dump and result files. The `read_spec` is used to match the infix from which dumps might need to be read. 
