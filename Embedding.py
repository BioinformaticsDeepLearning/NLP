# Apply Embedding methods#
 
# importing libraries
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec
 
#  Upload Data
sample = open("C:\\Users\\Alisha\\Desktop\\vig.csv")
sa = sample.read()

fm = sa.replace("\n", " ")
 
data = []
 
# iterate through each sentence in the file
for i in sent_tokenize(fm):
    temp = []
     
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
 
   data.append(temp)
 
1. # For embedding use CBOW#
model1 = gensim.models.Word2Vec(data, min_count = 1,
                              vector_size = 100, window = 5)
 
 
2. # For embedding use SkipGram#
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
                                             window = 5, sg = 1)
