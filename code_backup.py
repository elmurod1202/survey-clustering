# an example code for visualizing word embeddings in a 2D space.
# This code was inspired by a tutorial: https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5
# pretrained word embeddings for Spanish language was downloaded from: https://github.com/dccuchile/spanish-word-embeddings (the biggest one was taken)
# Author: Elmurod Kuriyozov (elmurod1202@gmail.com)
# Date: March 25, 2022

# import io
from sklearn.decomposition import PCA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
# from nltk.tokenize import word_tokenize
from smart_open import open, smart_open
import time

# Tried using cache functions, so I don't have to load the word-embedding model every time I run the code:
from functools import lru_cache

# This helps to find the "elbow point" on an optimization curve:
# Run this if you don't have it:
# $ pip install kneed
from kneed import KneeLocator

import json

# Just going to ignore warnings for a clear output, not recommended though:
import warnings
warnings.filterwarnings("ignore")


# Plots words in a given list based on PCA axes

# This is an old implementation, later replaced with gensim.
# def load_vectors(fname):
#     fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data

@lru_cache(maxsize=None)
def load_model(modelf):
    model = KeyedVectors.load_word2vec_format(modelf,binary=False)
    return model


# Model filename: This time it's Spanish pretrained word embedding.
modelf = "embeddings-l-model.vec"
# modelf = "sample_emb.vec"

# Name of file with words to be plotted, each word in a new line:
wordf = "result_words.txt"

# PCA axes to plot on, the most relevant are [0,1] or [1,2] for 2D and [0,1,2] or [1,2,3] for 3D
axes = [1, 2]

# Number of groups to cluster into, if <= 0 groups are instead created for each line in wordf
clusterK = 4


# To differentiate groups in the graph, you can give the labels a corresponding color or font size
# e.g. words in the first group will be red, words in the second group will be blue, etc.

# Color of words in each group, uses default if too many groups
# Dark colors are good for matplotlib's white background, use hex or https://matplotlib.org/gallery/color/named_colors.html
colors = ["tab:red", "tab:blue", "tab:green", "tab:orange",
          "tab:purple", "tab:olive", "tab:pink", "tab:cyan", "tab:gray", "tab:lime", "tab:brown", "tab:yellow"]
defaultcolor = "black"

# Font sizes of words in each group
sizes = []
defaultsize = 6


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
    
    
    # Uncomment these lines if you want to see the created plot:
    pyplot.plot(range(1, max_number_possible_clusters), wcss)
    pyplot.title('Elbow Method')
    pyplot.xlabel('Number of clusters')
    pyplot.ylabel('WCSS')
    pyplot.vlines(optimal_cluster_numbers, pyplot.ylim()[0], pyplot.ylim()[1], linestyles='dashed')
    #adding text inside the plot
    pyplot.text(optimal_cluster_numbers+1, 250, 'K='+str(optimal_cluster_numbers), fontsize = 16)
    result_filename = "output/elbow_method_2d.png"
    pyplot.savefig(result_filename)
    print("Elbow method:" + result_filename)
    pyplot.show()
    pyplot.close()

    return optimal_cluster_numbers

def plot2D_dots(result):
    pyplot.scatter(result[:, axes[0]], result[:, axes[1]], c="black", s=10)
    result_filename = "output/result_scatter_2d.png"
    pyplot.savefig(result_filename)
    print("2D output resulting scatter saved in file:" + result_filename)
    pyplot.close()


def plot2D(result, wordgroups):
    plot2D_dots(result)
    pyplot.scatter(result[:, axes[0]], result[:, axes[1]], c="black", s=10)
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            # Create plot point
            coord = (result[i, axes[0]], result[i, axes[1]])
            color = colors[g] if g < len(colors) else defaultcolor
            size = sizes[g] if g < len(sizes) else defaultsize
            pyplot.annotate(word, xy=coord, color=color, fontsize=size)
    result_filename = "output/result_words_2d.png"
    pyplot.savefig(result_filename)
    print("2D output resulting scatter with words saved in file:" + result_filename)
    pyplot.show()
    pyplot.close()


def plot3D(result, wordgroups):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, axes[0]], result[:, axes[1]], result[:, axes[2]])
    for g, group in enumerate(wordgroups):
        for word in group:
            if not word in words:
                continue
            i = words.index(word)
            # Create plot point
            color = colors[g] if g < len(colors) else defaultcolor
            size = sizes[g] if g < len(sizes) else defaultsize
            ax.text(result[i, axes[0]], result[i, axes[1]],
                    result[i, axes[2]], word, color=color, fontsize=size)
    result_filename = "output/result_words_3d.png"
    pyplot.savefig(result_filename)
    print("3D output result saved in file:" + result_filename)
    # pyplot.show()
    pyplot.close()


def get_words(wordf, model):
    words = []

    # # Extract words to plot from file
    # for line in open(wordf, "r", encoding="utf-8").read().split("\n"):
    #     l = [' '.join(word_tokenize(x)) for x in line.split(",")]
    #     l = filter(lambda x: x in model.wv.vocab.keys(), l)
    #     groups.append(l)
    #     words += l

    # Extract words to plot from file
    for word in open(wordf, "r", encoding="utf-8").read().split("\n"):
        # if (word in model.wv.vocab.keys()):
        if (word in list(model.index_to_key)):
            words.append(word)


    # Get word vectors from model
    vecs = {w: model.key_to_index[w] for w in words}

    return words, vecs

def get_groups(result, optimalK):
    groups = []

    # Assign groups if using clustering

    estimator = KMeans(init='k-means++', n_clusters=optimalK, n_init=10)
    estimator.fit_predict(model[vecs])
    groups = [[] for n in range(optimalK)]
    for i, w in enumerate(vecs.keys()):
        group = estimator.labels_[i]
        groups[group].append(w)

    return groups


if __name__ == '__main__':
    # Calculating the time it took to process everything:
    start_time = time.time()
    print("Programm started.")

    print("Loading the pretrained model:")
    # model = Word2Vec.load(modelf)
    
    model = load_model(modelf)

    print("Loading answer words from given file:")
    # Get groups by clustering
    words, vecs = get_words(wordf, model)

    coords = model[vecs]


    print("Plotting in 2D:")
    # Create 2D axes to plot on
    pca = PCA(n_components=max(axes)+1)
    result = pca.fit_transform(coords)

    # One problem that may arise is that you may need the optimal number of clusters, this can also be solved:
    # WCSS is defined as the sum of the squared distance between each member of the cluster and its centroid.
    # Now, let's try to find out the optimal number of clusters for the plot we have created using the elbow method.
    # To get the values used in the graph, we train multiple models using a different number of clusters
    #  and storing the value of the intertia_ property (WCSS) every time.
    # We graph the relationship between the number of clusters and Within Cluster Sum of Squares (WCSS),
    #  then we select the number of clusters where the change in WCSS begins to level off (elbow method).
    print("Obtaining the optimal cluster number:")
    optimalK = get_optimal_cluster_numbers(result)
    print("Obtained the optimal cluster number = {}".format(optimalK))
    
    # Now grouping the words:
    print("Grouping the words into clusters")
    groups = get_groups(vecs,optimalK)
    #Saving the resulting grouped data in a JSON format file:
    with open('result_groups.json', 'w') as f:
        json.dump(groups, f, indent=1)
        f.close()
    print("Grouped data saved in result_groups.json file.")
    with open('result_vectors.json', 'w') as f:
        for word_vec in list(result):
            json.dump(list(word_vec), f)
        f.close()
    print("resulting vectors data saved in result_vectors.json file.")


    print("Plotting clusters:")
    # Plot vectors on axes
    plot2D(result, groups)

    # You can also uncomment this part to plot the axes in 3D:
    # print("Plotting in 3D:")
    # # Create 3D axes to plot on
    # axes = [1, 2, 3]
    # pca = PCA(n_components=max(axes)+1)
    # result = pca.fit_transform(coords)
    # plot3D(result, groups)


    print("Programm finished.")
    finish_time = time.time()
    exec_time = finish_time - start_time
    print("It took {} seconds to execute this.".format(round(exec_time)))

