import warnings
import string
from nltk.tokenize import  word_tokenize,sent_tokenize
import re
from nltk.corpus import stopwords

from nltk.stem.porter import  PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


warnings.filterwarnings('ignore')

raw_docs = ["Turned 40 today. Covid ruined all my wife’s surprises. Fuck Donald Trump. Do something nice for someone random. BLM and for the love of God register to vote. Xo"
    ,"In hindsight, these are also very good tips to avoid Coronavirus. Try it out, extroverts, it’s a win-win."
            ]


# step1:转小写
raw_docs = [doc.lower() for doc in raw_docs]
print(raw_docs)

# step2:分词
tokenized_docs = [word_tokenize(doc) for doc in raw_docs]
print(tokenized_docs)

sent_token = [sent_tokenize(doc) for doc in raw_docs]

# step3:去除标点
regex = re.compile('[%s]' % re.escape(string.punctuation))

tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'',token)
        if not new_token == u'':
            new_review.append(new_token)
    tokenized_docs_no_punctuation.append(new_review)

print(tokenized_docs_no_punctuation)

# step4:去除停顿词
tokenized_docs_no_stopwords = []
for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)

    tokenized_docs_no_stopwords.append(new_term_vector)

print(tokenized_docs_no_stopwords)

# step5:词干化 词法化
porter = PorterStemmer()
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        # final_doc.append(porter.stem(word))
        final_doc.append(wordnet.lemmatize(word))
    preprocessed_docs.append(final_doc)

print(preprocessed_docs)

str_lists = []

# step6:将单词列表重新拼接
for i in range(0,len(raw_docs),1):
    str = ' '
    str_lists.append(str.join(preprocessed_docs[i]))

print(str_lists)

