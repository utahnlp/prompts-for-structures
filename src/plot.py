import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


