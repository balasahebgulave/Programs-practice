import nltk
from nltk.corpus import wordnet     #Import wordnet from the NLTK
first_word = wordnet.synset("organization.n.01")
second_word = wordnet.synset("company.n.01")
print('Similarity: ' + str(first_word.wup_similarity(second_word)))
