LDA Topic Modelling Dashboard
====================================================

This repo aims to build on the work of ``Cornell jsLDA`` and ``pyLDAvis`` by providing some added functionalities to allow for better understanding of topics within a document corpus. 

## Getting Started

The dashboard is currently not yet deployed or dockerized. For now, the instructions below are for running the dashboard on your local machine 

#### 1 In your shell, change the current directory to osprey_admin, where the Django code is 
```
cd osprey_admin 
```
#### 2 Install required packages 
```
pip3 install -r requirements.txt
```
#### 3 Run Django on local server
```
python manage.py runserver 8000
```
#### 4 Open on http://127.0.0.1:8000/ to view the dashboard

#### 5 Running the Dashboard Model 

a. Select the number of topics to train the LDA model with. \
b. Enter in Column to Size, the exact name of the column that we want to size with the LDA algorithm. \
c. Enter in Defining Column ID, the column that defines each unique value in Column to Size \
d. Upload the CSV and run the model. 

<br>

#### Click on the various bubbles to navigate the different topics. 

![alt-text](images/example_runtime.gif)

## Notes

Charts are coded in [D3.js](https://d3js.org/) and [Chart.js](https://www.chartjs.org/). Backend calculations are done in [DJango](https://www.djangoproject.com/). Latent Dirichlet allocation is described in [Blei et al. (2003)](https://jmlr.org/papers/v3/blei03a.html) and [Pritchard et al. (2000)](https://www.genetics.org/content/155/2/945). Intertopic Distances are calculated as described in [LDAvis](https://www.researchgate.net/profile/Carson-Sievert-2/publication/265784473_LDAvis_A_method_for_visualizing_and_interpreting_topics/links/541affaf0cf25ebee988df72/LDAvis-A-method-for-visualizing-and-interpreting-topics.pdf), using Jenson Shannon Divergence between topic words [Endres, D. M.; J. E. Schindelin (2003)](https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/1591/Endres2003-IEEETransInfTheory49-NewMetric.pdf;jsessionid=90CE76AB3BAA7256FC863BD493D8290F?sequence=1) and scaling the multi-dimensional data onto 2 dimensions using Principle Component Analysis (PCA) [Pearson K (1901)](https://en.wikipedia.org/wiki/Principal_component_analysis#cite_note-7)

** Design of the dashboard in the .gif might lag behind the current status of the project and is not reflective of the current state

## Upcoming Patches and Features 
Upcoming features, design plans and progress can be found on my notion page [here](https://www.notion.so/yikuang/944f4a6b77dc408c828d89842109599b?v=8b2a517d720b430b9b6ea4547594ccd3) 

## Other Implementations

* pyLDAvis [here](https://github.com/bmabey/pyLDAvis)
* Cornell jsLDA website [here](https://mimno.infosci.cornell.edu/jsLDA/jslda.html)
