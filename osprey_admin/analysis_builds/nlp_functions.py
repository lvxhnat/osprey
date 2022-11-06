import pandas as pd 
import numpy as np 
import sklearn

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

import os 
import collections

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

import osprey_admin.settings as settings

pd.options.mode.chained_assignment = None  # default='warn'

stopwords_path = os.path.join(settings.BASE_DIR,"analysis_builds/stopwords.csv")
stopwords = set(pd.read_csv(stopwords_path).stopwords)

def get_bigrams(dataframe, 
                num_topics: int,
               column_to_size: str = "tokenized_words",
               defining_class: str = "modularity_class") -> "DataFrame":
    
    dataframe = dataframe[[column_to_size, defining_class]]
    
    def bigramsfunc(x):
        '''
        Get the Bigrams given a string, with statistical implementation to get the bigrams with best associations
        '''
        try:
            bigram_finder = BigramCollocationFinder.from_words(x)
            # The arguments constitute the marginals of a contingency table, counting the occurrences of particular events in a corpus
            # Score the bigrams using Chi Square 
            bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 10)
            return [" ".join(bigram_tuple) for bigram_tuple in bigrams]
        except Exception as e:
            return None
    
    ### Remove words that have less than 4 characters per word 
    dataframe['bigrams'] = dataframe[column_to_size].str.join(" ").str.findall('\w{4,}').map(bigramsfunc)
    ### Concatenate by modularity class, then do a count by the number of items in np array, convert to DF and get top 1K words
    data_send = []
    for i in range(num_topics):
        data = pd.DataFrame.from_dict(collections.Counter(np.concatenate(dataframe[dataframe[defining_class] == i].bigrams.dropna().values)), orient = "index").sort_values(0, ascending = False).head(500)
        data['modularity_class'] = i
        data_send.append(data)
    
    return data_send

def get_topwords(dataframe, 
                 num_topics: int,
                 column_to_size: str = "tokenized_words", 
                 defining_class: str = "modularity_class") -> "DataFrame":
    '''
    Top 1000 words by modularity class 
    
    Average Run Time (100) on 35K rows: 0.453s. 
    
    Parameters
    ---------------------------------------------
    dataframe: dataframe; Pandas dataframe containing column_to_size and defining_class
    column_to_size: string: the column name we wish to size the top words for, it accepts a column containing a list of tokenized strings
    defining_class: string; the column name that defines the cluster/topics
    class_to_size: string; the class we want to size for, eg modularity class 2, we will size only within that 

    Returns
    ---------------------------------------------
    top50_all: dataframe; dataframe containing the top 50 tokenized words & respective frequency 
    top50_bymodclass: list; list of dataframe containing the top 50 tokenized words by modularity class & respective frequency
    all_topwords: dataframe; dataframe containing the top 1000 tokenized words by modularity class & respective frequency 
    '''
    topwordsby_class = []

    for mci in range(num_topics):
        mcidata = pd.DataFrame(np.concatenate(dataframe[dataframe[defining_class] == mci][column_to_size].values))[0].value_counts().reset_index().rename(columns = {0:"counts","index":"words"}).head(500)
        mcidata[defining_class] = mci
        topwordsby_class.append(mcidata.set_index("words"))

    ### Get the entire dataframe for sending to user
    all_topwords = pd.concat(topwordsby_class)

    ### Sort the values
    top50_all = all_topwords.drop(columns = [defining_class]).groupby("words").sum().reset_index().sort_values("counts", ascending = False).head(50)
    assert(len(top50_all) == 50)

    bag_of_words = pd.concat([topwordsby_class[i].drop(columns = ['modularity_class']).rename(columns = {"counts":str(i)}) for i in range(num_topics)], axis = 1).reset_index()
    bag_of_words = bag_of_words.rename(columns = {"index":"words"})
    bag_of_words = bag_of_words[bag_of_words.words.isin(top50_all.words)].fillna(0)

    top50_bymodclass = [bag_of_words[['words',str(i)]].rename(columns = {str(i):"counts"}) for i in range(num_topics)]
    
    return top50_all, top50_bymodclass, all_topwords

def get_tfidf_top_words(dataframe, 
                        default_string_column: str = "post_content",
                        column_to_size: str = "tokenized_words",
                       defining_class: str = "modularity_class") -> "DataFrame":
    r'''    
    Parameters
    ---------------------------------------------
    dataframe: dataframe; Pandas dataframe containing col_name_to_tokenize and defining_class
    default_string_column: string; the initial column name containing the strings that was used to sized for LDA
    col_name_to_tokenize: string; string of col name
    defining_class: string; string of col name
    
    Returns
    ---------------------------------------------
    top_words_tfidf: dataframe; dataframe containing the tokenized words, TF-IDF scores and the defining class it belongs to,
    denoted by the column name: tfidf_score_modclass{mod_class_id}
    '''    
        
    documents = [" ".join(dataframe[dataframe[defining_class] == i][default_string_column]) for i in dataframe[defining_class].unique()]

    dictionary = Dictionary([list(np.concatenate(dataframe[dataframe[defining_class] == i][column_to_size].values)) for i in dataframe[defining_class].unique()])
    
    # Create BoW corpus
    corpus = [dictionary.doc2bow(simple_preprocess(line)) for line in documents]
    # Create the TFIDF matrix
    tfidf = TfidfModel(corpus, smartirs='ntc')

    data = []
    for i in range(len(tfidf[corpus])):
        # Change to dataframe and map the words back 
        modclass_df = pd.DataFrame(tfidf[corpus][i], columns = ['word',str(i)]).sort_values(str(i))
        modclass_df['word'] = modclass_df['word'].apply(lambda x: dictionary[x])
        data.append(modclass_df.set_index('word'))
    
    # Get the max tfidf score per column and corresponding modularity class, then merge the two
    df1 = pd.concat(data, axis = 1).max(axis = 1).reset_index().rename(columns = {0:"score", "index":"word"})
    df2 = pd.concat(data, axis = 1).idxmax(axis = 1).reset_index().rename(columns = {0:"modularity_class", "index":"word"})

    top_words_tfidf = pd.merge(df1, df2, on = 'word', how = 'left')
    top500_words_tfidf = top_words_tfidf.sort_values(["modularity_class","score"], ascending = False).groupby("modularity_class").head(500)
    return top500_words_tfidf