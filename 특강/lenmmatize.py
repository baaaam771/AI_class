import nltk
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')


n = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going',
         'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

[n.lemmatize(w) for w in words]
