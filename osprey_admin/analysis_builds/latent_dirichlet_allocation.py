import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

import string 
import pandas as pd 
import numpy as np 
from operator import itemgetter
import warnings

import os
import re
import nltk
import osprey_admin.settings as settings

from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

stopwords_path = os.path.join(settings.BASE_DIR,"analysis_builds/stopwords.csv")
pd.options.mode.chained_assignment = None  # default='warn'

warnings.filterwarnings("ignore", category = DeprecationWarning)

### Removal of stopwords from set is much faster than from list 
stopwords = set(pd.read_csv(stopwords_path).stopwords)

nltk.download("punkt")
nltk.download("wordnet")

def LDA(dataframe, 
        column_to_size: str = "post_content",
        column_ids: str = "post_id",
        num_topics: int = 5,
        passes: int = 5) -> "Dataframe":
    
    """
    Runs Latent Dirichlet Analysis using the Gensim model, trained on text corpus passed into the function. 

    Parameters
    ---------------------------------------------
    dataframe : dataframe; minimally containing text to be sized and the respective IDs (This allows us to map the return dataframe back to initial)
    column_to_size: string; the column name containing the text corpus we wish to run topic modelling on 
    column_ids: string; the column name containing unique IDs identifiying unique strings within column_to_size
    num_topics: integer; the number of topics we wish to separate from initial text corpus 
    passes: integer; the number of iterations to run the LDA model on the text corpus 

    Returns
    ---------------------------------------------
    tagged_LDA : dataframe; containing the original text corpus with the unique IDs, the respective modularity class it has been mapped to, and the confidence, P(In mod class N) of the label
    [New] added in a column for tokenzied words
    lda_model : gensim LDA object 
    formula : returns a list of strings containing the top words in each topic
    """
    
    # Get necessary post content only 
    papers = dataframe[[column_to_size, column_ids]].drop_duplicates()
    ## Remove punctuation
    papers['cleaned_sentence'] = papers[column_to_size].map(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)).replace('’',"").replace("”","").replace("“",""))
    ## Clean string to avoid malformed csvs
    papers['cleaned_sentence'] = papers['cleaned_sentence'].str.replace('\n', ' // ').str.replace('\t','').str.replace('\r','')
    ### Data Processing 
    # Tokenization, lemmatization, url, number and stopword removal 
    papers['tokenized_words'] = papers['cleaned_sentence'].apply(lambda x: [i for i in nltk.word_tokenize(x) if i not in stopwords and "https" not in i and not i.isdigit()])

    data_words = papers['tokenized_words'].values.tolist()

    # Create Dictionary, mapping the words to a particular ID
    id2word = corpora.Dictionary(data_words)
    # Map the words back to their assigned ID and the frequency of the word. This returns a list of tuples [(Word ID, Frequency)]
    corpus = [id2word.doc2bow(text) for text in data_words]
    
    # Train models via multithreading 
    # # https://radimrehurek.com/gensim/models/ldamulticore.html
    ## Sets a random state to ensure reproducible results
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id2word, num_topics = num_topics, passes = 5, random_state = np.random.RandomState(1234))
    
    # Return the topics that each document is tagged too and the probability scores for each of them 
    # Takes in a Bag Of Words
    assigned_documents = lda_model.get_document_topics(corpus)
    
    # Get the document with the highest probability of being tagged with the cluster 
    fis = [max(item,key = itemgetter(1)) for item in assigned_documents]
    text_and_tags = pd.DataFrame(fis, columns = ['modularity_class', 'probability_score'])
    # Format final output 
    text_and_tags.modularity_class = text_and_tags.modularity_class.astype(int)
    
    papers.reset_index(drop = True, inplace = True)
    tagged_LDA = pd.merge(papers.reset_index(), text_and_tags.reset_index(), on = 'index', how = 'left')
    
    return tagged_LDA, lda_model, [" ".join(re.findall('"(.*?)"', i[1])) for i in lda_model.print_topics()]



def intertopic_distance(LDA_dataframe, LDAModel, defining_class: str = "modularity_class"):
    '''
    Calculate the intertopic distance using jenson shannon distance and PCA for dimension reduction 
    '''
    ### Get the probability scores for each topic for each word in the entire document 
    LDA_topics = LDAModel.get_topics()
    ### Set the number of components to 2 i.e, reduce number of dimensions to 2
    pca = PCA(n_components=2)

    topic_size = LDA_dataframe[defining_class].value_counts().values
    
    dats = []
    ### Run this in the format of a matrix
    for i in range(len(LDA_topics)):
        row = []
        for a in range(len(LDA_topics)):
            ### Jenson shannon distance to calculate information entropy 
            row.append(distance.jensenshannon(LDA_topics[i], LDA_topics[a]))
        dats.append(np.array(row))
    dats = np.array(dats)
    dats = pca.fit_transform(dats)
    
    ### Min max normalize the data
    mins, maxs = topic_size.min(), topic_size.max()
    def minmaxnorm(x):
        return ((x - mins)/(maxs - mins)) + 0.01

    topic_size = np.array(list(map(minmaxnorm, topic_size)))

    final_dict = {}
    for i in range(len(dats)):
        data_to_app = [topic_size[i] * 10] + list(dats[i])
        final_dict[i] = data_to_app
    
    return topic_size, final_dict