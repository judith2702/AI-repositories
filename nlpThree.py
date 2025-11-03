from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC 
import spacy
import nltk

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')



corpus = [
    "the movie was fantastic and i loved every part of it",
    "an absolute masterpiece with brilliant acting",
    "the film was boring and too long",
    "i really enjoyed the story and the visuals",
    "the plot was terrible and the acting was even worse",
    "what a wonderful experience, highly recommend",
    "not worth my time, very disappointing",
    "a truly great film, i will watch it again",
    "the script was weak and the characters were flat",
    "an amazing journey from start to finish",
    "lovelly film with good actors",
    "Bad acting just one time watch"
]

Positive = 'positive'
Negative = 'negative'


categories = [
    "Positive", "Positive", "Negative", "Positive", "Negative",
    "Positive", "Negative", "Positive", "Negative", "Positive", 
    "Positive","Negative"
]

vectorizer = CountVectorizer(ngram_range=(1,2))

vectors = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(vectors.toarray())

clf = SVC(kernel = 'linear')
clf.fit(vectors, categories)
test_corpus = [
    "the movie was great",
    "i hated the film",
    "a boring and bad story",
    "absolutely loved it"
]
test_categories = [nlp(text) for text in test_corpus]
test_x = vectorizer.transform(test_corpus)
print(clf.predict(test_x))



