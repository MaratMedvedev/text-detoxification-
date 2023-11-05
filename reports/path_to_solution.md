# Dataset
My exploration of dataset store here `notebooks/data_exploration_and_preprocessing.ipynb`.

Dataset that was given is a little strange. It was conceived as a dataset that stores paraphrased sentences with less toxicity
But in some cases, reference text has less toxicity than translation text. So, I decide just swap two datapoints values 
if toxicity of first is less than of the second to correct logic of dataset.

# Metrics
Firstly, I think about what metrics should I use to understand quality of my model.

I need two metrics: similarity of two texts and toxicity of text.

My exploration of similarity metric store here `notebooks/Exploration_approaches_for_estimating_sentences_similarity.ipynb`.

For text similarity, I decide to use cosine similarity between embeddings of pretrained Hugging faces [transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
You can see this function in `src/models/evaluate.py` texts_similarity function.

My exploration of similarity metric store here `Exploration_estimating_sentence_toxification.ipynb`.

For text toxicity, I decide use pretrained Hugging faces [transformer](https://huggingface.co/cointegrated/rubert-tiny-toxicity?text=You+fucking+idiot%21).
But this transformer take text and return five values.
Each value is probability of some toxicity class: insult

1. insult
2. dangerous
3. non-toxic
4. threat
5. obscenity

I just accumulate all probabilities in one probability of toxicity.
You can see this function in `src/models/evaluate.py` text_toxicity function.

# Model

First, I create baseline model that just return that take. 
This model is `models/baseline.py`

Second, I create model that delete all words that contain swear words.
Swear words I get from [this](http://www.bannedwordlist.com/) dataset.
This dataset located in `data/external/swearWords.csv`.
This model is `models/delete_bad_words_model.py`

Third, I try to use some pretrained model to get detoxicate text. 
My exploration you can see here `notebooks/solution_exploration.ipynb`.
After, t5-small model doesn't give some good results, I decide to try their big brother
t5-base model. Exploration this model you can see here `notebooks/fine_tune_t5_for_detoxication.ipynb`.

But this model trained about 2 hours(on Kaggles GPU accelerators) and it just learns output the same text...
And you need 2.38 GB to store this model. So, it very bad solution.





# Model evalution
Code for models evaluation here `src/visualization/visualize.py`

| Model                    | Average similarity | Average toxicity |
|--------------------------|--------------------|------------------|
| baseline                 | 1.0                | 0.518            |        |
| deleting bad words model | 0.981              | 0.494            |