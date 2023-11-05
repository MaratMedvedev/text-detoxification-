import pandas as pd


class Model:
    def __init__(self, path_to_bad_words):
        self.bad_words = list(pd.read_csv(path_to_bad_words, sep=','))
    def predict(self, texts):
        ans = []
        for text in texts:
            splt_text = text.split()
            new_text = []
            for i in range(len(splt_text)):
                if not (splt_text[i].lower() in self.bad_words):
                    new_text.append(splt_text[i])
            ans.append(' '.join(new_text))
        return ans