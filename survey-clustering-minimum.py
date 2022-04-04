# A minimized code for clustering texts by their meanings. 
# For the complete code with visualised giagrams, please use the survey-clustering.py file, but it requires more libraries/frameworks to be run.
# Pretrained word embeddings for Spanish language was downloaded from: https://github.com/dccuchile/spanish-word-embeddings (the biggest one was taken)
# Author: Elmurod Kuriyozov (elmurod1202@gmail.com)
# Date: April 5, 2022

from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
# This helps to find the "elbow point" on an optimization curve:
from kneed import KneeLocator
import numpy as np
import csv

# Model filename: This time it's Spanish pretrained word embedding.
modelf = "src/embeddings-l-model.vec"

# Name of file with words to be plotted, each word in a new line:
wordf = "input/answers.txt"

# Name of the file that holds the list of Spanish stopwords, one word per line.
stopwords_file = "src/spanish-stopwords.txt"

# Number of groups/clusters to split the texts into. 
# Keep it 0 if you don't have an exact number and the code generates the optimal number for the given texts.
clusterK = 0

# A function to return a list of words read from a given file:
def load_model(modelf):
    model = KeyedVectors.load_word2vec_format(modelf,binary=False)
    return model

# A method to return a set of stopwords read from a given file:
def load_stopwords(file_name):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    file.close()
    return set(lines)

# A function to return average vector of given vectors:
def makeFeatureVec(words, model):
    # Function to average all of the word vectors in a given expression
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((300,),dtype="float32")
    nwords = 0.
    # Loop over each word and add its feature vector to the total
    for word in words:
        if (word in list(model.index_to_key)):
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
        else:
            print("!!! Alert: OOV: ", word)

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


# A function to remove stopwords from givem multi-word expression:
def remove_stopwords(multiword, stopwords):
    important_words =[]
    for word in multiword:
        if word not in stopwords:
            important_words.append(word)
        
    return important_words

# A function that finds the optimal numbeer of clusters using the elbow method:
def get_optimal_cluster_numbers(result):
    max_number_possible_clusters = 30
    optimal_cluster_numbers = 0
    wcss = []
    for i in range(1, max_number_possible_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(result)
        wcss.append(kmeans.inertia_)
    
    # Now let's find out the maximum curvation point using elbow method:
    # For this we used an implementation: https://github.com/arvkevi/kneed
    kneedle = KneeLocator(range(1, max_number_possible_clusters), wcss, curve="convex", direction="decreasing")
    optimal_cluster_numbers = kneedle.elbow
    
    return optimal_cluster_numbers

# A function that reads the texts from a given file, obtains their vectors and returns it
def get_words(wordf, model, stopwords):
    words = []

    # Extract words to plot from file
    for word in open(wordf, "r", encoding="utf-8").read().split("\n"):
        # if (word in list(model.index_to_key)):
        if len(word) > 0:
            words.append(word)

    # Get word vectors from model
    # vecs = {w: model.key_to_index[w] for w in words}
    vecs = {}
    words_new = []
    for count_word, word in enumerate(words):
        vec_done = False
        vec = 0
        if "_" not in word:
            # This means it's a single word, so we get only its vector:
            if (word in list(model.index_to_key)):
                vecs[word] = model.key_to_index[word]
                vec = model.key_to_index[word]
                vec_done = True
                words_new.append(word)
            else:
                print("!!! Alert: OOV: ", word)

        else:
            # This means the expression is not a single word
            # We have to deal with this carefully:
            # First we split it to separate words, then remove stopwords:
            expression = word.split("_")
            expression_chunks = remove_stopwords(expression, stopwords)
            print("Found a stopword: {}, un-stopworded it:{}".format(word, ' '.join(expression_chunks)))
            if len(expression_chunks) == 0:
                print("!!! Alert: OOV, could not handle un-stopwording: ", word)
            if len(expression_chunks) == 1:
                # This means after removing stopword it became a single word
                if (expression_chunks[0] in list(model.index_to_key)):
                    new_vec = model[expression_chunks[0]]
                    # We also now have to add this expression to the list of vectors in the model:
                    new_vec_index = model.add_vector(word, new_vec)
                    print("new-vec_index:", new_vec_index)
                    vecs[word] = new_vec_index
                    vec = new_vec_index
                    vec_done = True
                    words_new.append(word)
                else:
                    print("!!! Alert: OOV: ", word)
            else:
                # This means that it still consists of multiple words
                # So we consider their average vector for that:
                new_vec_index = model.add_vector(word, makeFeatureVec(expression_chunks, model))
                vecs[word] = new_vec_index
                print("new-vec_index:", new_vec_index)
                vec_done = True
                vec = new_vec_index
                words_new.append(word)
                # We also now have to add this expression to the list of vectors in the model:
        if not vec_done:
            print("!!!!!!!!!!!!! One word could not get a vector:", word)

    return words_new, vecs, model

# A function that splits given wordlist into clusters of k numbers
def get_groups(vecs, optimalK, model):
    groups = []

    # Assign groups if using clustering
    estimator = KMeans(init='k-means++', n_clusters=optimalK, n_init=10)
    estimator.fit_predict(model[vecs])
    groups = [[] for n in range(optimalK)]
    for i, w in enumerate(vecs.keys()):
        group = estimator.labels_[i]
        groups[group].append(w)

    return groups

# The main function that runs the whole program:
if __name__ == '__main__':
    print("Programm started.")

    print("Loading the pretrained model:")
    print("Please wait, it usually takes a few minutes...")
    model = load_model(modelf)

    print("Loading stopwords:")
    stopwords = load_stopwords(stopwords_file)

    print("Loading answer words from given file:")
    words, vecs, model_new = get_words(wordf, model, stopwords)

    coords = model_new[vecs]
    print("Number of Words: ", len(words))
   
    # Reducing the vector dimensions down to two:
    pca = PCA(n_components=2)
    result = pca.fit_transform(coords)

    # Dividing into groups/ Clustering:
    if clusterK == 0:
        # WCSS is defined as the sum of the squared distance between each member of the cluster and its centroid.
        # We graph the relationship between the number of clusters and Within Cluster Sum of Squares (WCSS),
        #  then we select the number of clusters where the change in WCSS begins to level off (elbow method).
        print("Obtaining the optimal cluster number:")
        optimalK = get_optimal_cluster_numbers(result)
        print("Obtained the optimal cluster number = {}".format(optimalK))
        clusterK = optimalK
    else:
        print("Using given number of clusters: ", clusterK)

    # Now grouping the words:
    print("Grouping the words into clusters")
    groups = get_groups(vecs,clusterK, model_new)
    
   
    # Saving the resulting vectors in a csv file:
    result_file_name = "output/result.csv"
    with open(result_file_name, 'w', encoding='UTF8') as out:
        # create the csv writer
        writer = csv.writer(out, dialect='excel')
        header = ['word', 'group', 'vector_x', 'vector_y']
        # write the header
        writer.writerow(header)
        # Loop through each word in groups and write them to the file:
        for g, group in enumerate(groups):
            group_id = g+1
            for word in group:
                if not word in words:
                    continue
                i = words.index(word)
                data_row = [word, group_id, result[i, 0], result[i, 1]]
                # write a row to the csv file
                writer.writerow(data_row)
    out.close()

    print("Resulting data saved in a file: ", result_file_name)

    # Done.
    print("Programm finished.")
