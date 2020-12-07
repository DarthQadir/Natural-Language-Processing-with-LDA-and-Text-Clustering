<h3 align="center">
This repository serves as a guide for LDA, Semantic Similarity, and Text Clustering applied to <a href="https://www.kaggle.com/c/data-science-for-good-careervillage"> CareerVillage's kaggle challenge </a>.
  <br></br>
</h3>

**An article that goes into depth about all these steps and the notebook can be found here (will put link once others have uploaded their medium article, and my own medium article is published)**

<br>

The Jupyter Notebook in the repository contains 3 sections:

1. Cloning the Git Repository with the data, and preprocessing the data for our work.
2. Performing LDA on a large number of questions, and answers.
3. Performing Clustering using Spacy and scikit-learn's DBSCAN clustering algorithm.


<br>

<h3 align='center'>  Cloning the Git Repository with the data, and preprocessing the data for our work </h3>

Run the cells below the title **"Clone Repo to get data and preprocess data"** all the way upto the cell titled **NLP Functions**

<br>

<h3 align='center'>  Performing LDA on a large number of questions, and answers </h3>

Latent Dirichlet Allocation (LDA) helps us find topics in the dataset of questions, and in the dataset of answers. We use the top two words that describe these topics as new features/tags for each professional. 

*Function that performs LDA* 

```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

unique_tags = set(tags['tags_tag_name'])

def find_topics(question_body):
  """
  Function that takes a question as an input, and finds the two most important topics/tags
  If the found topics exist in the already existing database of tags, we add these tags
  to the professional who answered the question
  """
  try:
    text = nlp_pipeline(question_body)
    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform([text])
    number_topics = 1
    number_words = 2
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    words = count_vectorizer.get_feature_names()
    topics = [[words[i] for i in topic.argsort()[:-number_words - 1:-1]] for (topic_idx, topic) in enumerate(lda.components_)]
    topics = np.array(topics).ravel()
    existing_topics = set.intersection(set(topics),unique_tags)
  
  except:
    print(question_body)
    return (question_body)

  return existing_topics
```

<h3 align='center'> Performing Clustering using Spacy and scikit-learn's DBSCAN clustering algorithm </h3>

<br>

We use Spacy's similarity model to map tags to a 300-dimensional vector space. Then we use scikit-learn's DBSCAN clustering model to find clusters within that vector space.

```
import en_core_web_lg
nlp = en_core_web_lg.load()

tag_list =  list(tags['tags_tag_name'])
corpus = ' '.join(list(tag_list)).replace('-',' ')
words = corpus.split()
corpus = " ".join(sorted(set(words), key=words.index))
tokens = nlp(corpus)

word_vectors = []
for i in tokens:
  word_vectors.append(i.vector)
word_vectors = np.array(word_vectors)

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(metric='cosine', eps=0.3, min_samples=2).fit(word_vectors)
```


