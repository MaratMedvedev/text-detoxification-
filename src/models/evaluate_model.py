def evaluate(model, data):
    results = model.predict(data)
    mean_toxicity = 0
    print("Model evaluating toxicity...")
    for text in results:
        mean_toxicity += text_toxicity(text)
    mean_toxicity /= len(results)

    print("Model evaluating text similarity...")
    mean_similarity = 0
    for i, text in enumerate(results):
        mean_similarity += texts_similarity(text, data[i])
    mean_similarity /= len(results)
    return mean_similarity, mean_toxicity


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def text_toxicity(text):
    model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    return 1 - proba.T[0] * (1 - proba.T[-1])

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
def texts_similarity(text1, text2):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_1 = model.encode([text1], convert_to_tensor=True)
    embedding_2 = model.encode([text2], convert_to_tensor=True)
    return cosine_similarity(embedding_1, embedding_2)[0][0]