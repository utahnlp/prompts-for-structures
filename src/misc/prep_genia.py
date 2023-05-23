import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil

DATA_FOLDER = "./../../data/GENIA_MedCo_coreference_corpus_1.0/"

xml_files = list(Path(DATA_FOLDER).glob("*.xml"))

files_train, files_devtest = train_test_split(xml_files, test_size=0.3, random_state=42, shuffle=True)
files_dev, files_test = train_test_split(files_devtest, test_size=0.5, shuffle=False)


def copy_files_to_split(split, files):
        folder_path = Path(DATA_FOLDER,split)
        folder_path.mkdir(exist_ok = True, parents = True)
        for f in files:
            shutil.copy(f, folder_path)


copy_files_to_split("train", files_train)
copy_files_to_split("dev", files_dev)
copy_files_to_split("test", files_test)

