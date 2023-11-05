# In this file, I evaluate each model by considering
import sys
sys.path.append('../data')
import make_dataset
data = list(make_dataset.make_dataset_toxic_text_to_neutral_text(100).iloc[:, 0])

import sys
sys.path.append('../../models')
import baseline
import delete_bad_words_model

bl = baseline.Model()
path_to_bad_words = "../../data/external/swearWords.csv"
del_bad_words = delete_bad_words_model.Model(path_to_bad_words)

import sys
sys.path.append('../models')
from evaluate_model import evaluate
sim_bl, tox_bl = evaluate(bl, data)
sim_del_bad_words, tox_del_bad_words = evaluate(del_bad_words, data))
print(f"Average similarity for baseline model: {sim_bl}, average toxicity for baseline model: {tox_bl}")
print(f"Average similarity for delete bad words model: {sim_del_bad_words}, average toxicity for delete bad words  model: {tox_del_bad_words}") # (0.981007696390152, 0.49400427074495484)
