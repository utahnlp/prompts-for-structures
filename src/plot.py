import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

sns.set_theme(style='whitegrid')


def plot_yes_no(gold, generations, prefix=''):
    """
    Inputs
    ---------------
    gold - List[str]. List of gold answers
    generations - List[dict]. List of Yes/No generations with scores
    prefix - str. Prefix to add to the filename. Like an identifier.
    """
    df_data  = []
    for ix, g_el in enumerate(gold):
        update = [g_el, 0, 0]
        for gen_el in generations[ix]:
            if gen_el['sentence'] == 'No':
                update[1] = gen_el['score']
            elif gen_el['sentence'] == 'Yes':
                update[2] = gen_el['score']
        df_data.append(update)

    data = pd.DataFrame(df_data, columns=['gold', 'no_prob', 'yes_prob'])

    plot = sns.kdeplot(data=data, x="yes_prob", hue="gold", multiple="stack")
    plot.set_xlim(0,1)
    plot.set_title(f"Probability of Generating \'Yes\' w.r.t Gold Labels")
    plot.set_xlabel("\'Yes\' Generation Probability")
    #plot.set_ylabel("")filename = "_".join(lab.lower().split())
    plot.figure.savefig(f"./../figures/{prefix}yes_probab.pdf")




def plot_score_diff(gold, generations, prefix=''):
    """
    Inputs
    ---------------
    gold - List[str]. List of gold answers
    generations - List[dict]. List of Yes/No generations with scores
    prefix - str. Prefix to add to the filename. Like an identifier.
    """
    df_data  = []
    for ix, g_el in enumerate(gold):
        update = [g_el, 0]
        for gen_el in generations[ix]:
            if gen_el['sentence'] == 'No':
                update[1] -= gen_el['score']
            elif gen_el['sentence'] == 'Yes':
                update[1] += gen_el['score']
        df_data.append(update)

    data = pd.DataFrame(df_data, columns=['gold', 'score_diff'])

    plot = sns.histplot(data=data, x="score_diff", hue="gold", multiple="stack", kde=True)
    plot.set_title(f"Difference Scores \'Yes\' and \'No\'")
    plot.set_xlabel("Score Difference")
    #plot.set_ylabel("")filename = "_".join(lab.lower().split())
    plot.figure.savefig(f"./../figures/{prefix}score_diff.pdf")






def plot_calibration(gens, gold, filename):
    frac = []
    i= 0
    while i <=1:
        frac.append(i)
        i+=0.05
    
    counts = []
    mae = 0
    for f in frac:
        frac_count = 0
        yes_cnt = 0
        for ix, gen in enumerate(gens): 
            if gold[ix] == "Yes":
                yes_cnt += 1
            for g in gen:
                if g['sentence'] == 'Yes':
                
                    if g["score"] <= f and gold[ix]=="Yes":
                        frac_count += 1
        data_p = frac_count/yes_cnt
        
        mae += abs(data_p - f)/math.sqrt(2)

        counts.append(data_p)
   
    plot = sns.lineplot(data=pd.DataFrame({"pr": frac, "density":counts}), x="pr", y="density")
    plot.set_xlim(0,1)
    plot.set_ylim(0,1)
    plot.set_title(f"Calibration Plot (Pre-Calib)")
    plot.set_xlabel("Fraction (f)")
    plot.set_ylabel("Fraction of Instances Labeled 'Yes' with Pr('Yes') <= f")
    #plot.set_ylabel("")filename = "_".join(lab.lower().split())
    plot.figure.savefig(filename)
    print(mae/len(counts))

        
    
