from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.core.files.storage import FileSystemStorage

from analysis_builds.latent_dirichlet_allocation import LDA, intertopic_distance
from analysis_builds import nlp_functions 

import io
import os 
import csv
import zipfile

import logging 
import pandas as pd
from wsgiref.util import FileWrapper
from django.conf import settings
import mimetypes
import time 

logger = logging.getLogger(__name__)

# Create your views here.
def indexPage(request):

    if request.FILES.get("input_file"):   ### request.FILES in this case because input is a file, if input_file is not empty, run this
        a = time.time()

        # get the uploaded file
        my_uploaded_file = request.FILES.get('input_file')
        num_of_topics = request.POST.get("slider_control")
        column_to_size, column_ids = request.POST.get("col_to_size"), request.POST.get("defining_col_id")
        # Read as dataframe 
        dataframe = pd.read_csv(io.StringIO(my_uploaded_file.read().decode('utf-8')), delimiter=',')
        totalCount = dataframe.shape[0]
        # Modularity Class list 
        modclass_list = range(int(num_of_topics))

        ### Run Latent Dirichlet Analysis on the dataframe if a file is uploaded
        LDA_output, LDA_model, top_topic_words = LDA(dataframe, num_topics = num_of_topics, column_to_size = column_to_size, column_ids = column_ids)

        ### Get the top 50 words by word frequency x cluster 
        all_top50, modclass_top50, TOPWORDS = nlp_functions.get_topwords(LDA_output, num_topics = int(num_of_topics))
        
        all_top50 = all_top50.rename(columns = {"index": "words"})
        # All top 50 words (Baseline data)
        alltopwordslabels = all_top50.head(50).words.to_list()
        alltopwords = all_top50.head(50).counts.to_list()
        # All top 50 words per modularity cluster (By mod class data)
        topwords_data = [word_chunk.counts.to_list() for word_chunk in modclass_top50]

        ### Get the top 50 TF-IDF words by cluster 
        TFIDF = nlp_functions.get_tfidf_top_words(LDA_output, default_string_column = column_to_size)
        TFIDF_by_modclass = [TFIDF[TFIDF.modularity_class == str(i)].sort_values('score', ascending = False).head(50) for i in range(int(num_of_topics))]
        TFIDF_data = [chunk.score.to_list() for chunk in TFIDF_by_modclass]
        TFIDF_label = [chunk.word.to_list() for chunk in TFIDF_by_modclass]

        ### Get the top 50 Bi-Grams by overall x cluster 
        BIGRAMS = nlp_functions.get_bigrams(LDA_output, num_topics = int(num_of_topics))
        BIGRAMS_data = [chunk.head(50)[0].to_list() for chunk in BIGRAMS] # The column name is 0 
        BIGRAMS_label = [chunk.head(50).index.to_list() for chunk in BIGRAMS]
        BIGRAMS = pd.concat(BIGRAMS)
        
        topic_size, intertopic_coordinates = intertopic_distance(LDA_output, LDA_model)
        
        b = time.time()
        runTime = round(b - a, 2) 

        context = {'alltopwords':alltopwords,
        'alltopwordslabels':alltopwordslabels,
        'topwords_data':topwords_data,
        'modclass_list':modclass_list,
        'tfidf_label':TFIDF_label,
        'tfidf_data':TFIDF_data,
        'totalCount':totalCount,
        'runTime':runTime,
        'bigrams_label':BIGRAMS_label,
        'bigrams_data':BIGRAMS_data,
        'intertopic_distances':intertopic_coordinates,
        'topic_size':topic_size,
        'top_topic_words':top_topic_words,
        }

        final_downloads = {}

        with zipfile.ZipFile('static/OUTPUTS.zip', 'w') as csv_zip:
            csv_zip.writestr("LDA.csv", LDA_output.drop(columns = ['tokenized_words']).to_csv())
            csv_zip.writestr("Topwords.csv", TOPWORDS.reset_index().to_csv()) 
            csv_zip.writestr("TFIDF.csv", TFIDF.to_csv())
            csv_zip.writestr("Bi_Grams.csv", BIGRAMS.reset_index().rename(columns = {0:"word_counts","index":"bigram"}).to_csv())
            csv_zip.writestr("Stopwords.csv", pd.read_csv(os.path.join(os.path.dirname(settings.BASE_DIR), 'osprey_admin/analysis_builds/stopwords.csv')).to_csv())

        return render(request, 'index.html', context)

    else:
        return render(request, 'index.html')