import nltk
import re
import time

# Required NLTK downloads
nltk.download('stopwords')   # Downloads the list of stopwords.
nltk.download('punkt')  #Downloads the Punkt tokenizer models.
nltk.download('wordnet')  #Downloads the WordNet lexical database.is a large English dictionary database that groups English words into sets of synonyms
nltk.download('averaged_perceptron_tagger')  #Downloads the POS (Part-Of-Speech) tagger model.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Example text
text = "This is a simple example to demonstrate stopword removal using NLTK."

# Step 1: Tokenization and Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_texts = [word for word in word_tokenize(text) if word.lower() not in stop_words]
mywords = filtered_texts
print("\nFiltered Tokens:")
print(mywords)

# Step 2: Stemming
print("\nStemming")
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in mywords]
print("Stemmed Tokens:")
print(stemmed_words)

# Step 3: Lemmatization
print("\nLemmatization")
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in mywords]
print("Lemmatized Tokens:")
print(lemmatized_words)

# Step 4: POS Tagging
print("\nPOS Tagging")
pos_tags = pos_tag(lemmatized_words)
print("POS Tags:")
print(pos_tags)

# Step 5: Regex Cleaning
txt = " ".join(lemmatized_words)
cleaned_txt = re.sub('[^a-zA-Z0-9]', ' ', txt)
cleaned_txt = re.sub(' +', ' ', cleaned_txt).strip()
print("\nCleaned Text:")
print(cleaned_txt)

# Step 6: One-Hot Encoding
print("\nOne-Hot Encoding")
corpus = [cleaned_txt]
onehot_vectorizer = CountVectorizer(binary=True)
onehot_matrix = onehot_vectorizer.fit_transform(corpus)
print(onehot_matrix.toarray())
print("Feature Names:")
print(onehot_vectorizer.get_feature_names_out())

# Step 7: TF-IDF
print("\nTF-IDF")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print(tfidf_matrix.toarray())
print("TF-IDF Feature Names:")
print(tfidf_vectorizer.get_feature_names_out())

# Step 8: Word2Vec Embedding (Train your own)
print("\nWord2Vec Embedding (Trained)")
from gensim.models import Word2Vec
tokenized_text = [word_tokenize(text.lower())]
w2v_model = Word2Vec(sentences=tokenized_text, vector_size=50, window=5, min_count=1, workers=2)

for word in mywords:
    if word.lower() in w2v_model.wv:
        print(f"Word2Vec vector for '{word}':\n", w2v_model.wv[word.lower()])
    else:
        print(f"'{word}' not in Word2Vec vocabulary")

# Step 9: GloVe Embedding (Pretrained)
print("\nGloVe Embedding (Pretrained)")
import gensim.downloader as api
glove_model = api.load("glove-wiki-gigaword-100")

for word in mywords:
    if word.lower() in glove_model:
        print(f"GloVe vector for '{word}':\n", glove_model[word.lower()])
    else:
        print(f"'{word}' not in GloVe vocabulary")

# Step 10: FastText Embedding (Pretrained)
print("\nFastText Embedding (Pretrained)")
fasttext_model = api.load("fasttext-wiki-news-subwords-300")

for word in mywords:
    if word.lower() in fasttext_model:
        print(f"FastText vector for '{word}':\n", fasttext_model[word.lower()])
    else:
        print(f"'{word}' not in FastText vocabulary")

# Execution Time
start = time.time()
_ = sum([i for i in range(1000000)])
end = time.time()
print("\nExecution Time:", round(end - start, 4), "seconds")
