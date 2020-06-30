from urllib.request import urlopen
import nltk
from bs4 import BeautifulSoup
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

# PLACE CONTENTS OF HTML FILE IN BEAUTIFUL SOUP
url = "https://en.wikipedia.org/wiki/Google"
html = urlopen(url)
soup = BeautifulSoup(html, "html.parser")

# PLACE CONTENTS OF ALL <P> IN LIST AND THEN INPUT.TXT
content = []
for link in soup.find_all('p'):
    content.append(link.text)
file = open('input.txt', 'w+')
for item in content:
    file.write(item)
file.close()

# PLACE INPUT.TXT CONTENTS INTO A STRING
file = open('input.txt', 'r')
data_string = file.read()
data_string = data_string.strip('\n')
print("Original Data -> \n", data_string, '\n')

# TOKENIZATION
word_tokens = nltk.word_tokenize(data_string)
print("Tokens of Original Data -> \n", word_tokens, '\n')

# POS
word_pos = nltk.pos_tag(word_tokens)
print("Word POS -> \n", word_pos, '\n')

# STEMMING
pStemmer = PorterStemmer()
stem = ''
for word in word_tokens:
    stem = stem + pStemmer.stem(word) + ' '
print("Output from PorterStemmer -> \n", stem, '\n')

# LEMMATIZATION
lemmatizer = WordNetLemmatizer()
lemma = ''
for word in word_tokens:
    lemma = lemma + lemmatizer.lemmatize(word) + ' '
print("Output from lemmatization -> \n", lemma, '\n')

# TRIGRAM
trigram = ngrams(word_tokens, 3)
print("Output from trigram -> \n")
for word in trigram:
    print(word)
print()

# NER
ner = ne_chunk(pos_tag(wordpunct_tokenize(data_string)))
print("Output from NER -> \n", ner)
