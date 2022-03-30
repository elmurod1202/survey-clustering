# An example of stop-word removal in python.
# This code loads the list of Spanish stopwords from given file,
# also loads list of answers from another file and when it finds multiple-word expressions in an answer,
# the code tries to remove stopwords from it. (Useful for vector averaging)
# Author: Elmurod Kuriyozov (elmurod1202@gmail.com)
# Date: March 25, 2022


from smart_open import open
import time
from pprint import pprint
import numpy as np

# Name of file with words to be plotted, each word in a new line:
wordf = "input/answers.txt"

# Name of the file that holds the list of Spanish stopwords, one word per line.
stopwords_file = "src/spanish-stopwords.txt"


# A method to return a set of stopwords read from a given file:
def load_stopwords(file_name):
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]
    file.close()
    return set(lines)


# A function to remove stopwords from given multi-word expression:
def remove_stopwords(multiword, stopwords):
    important_words =[]
    for word in multiword:
        if word not in stopwords:
            important_words.append(word)
        
    return important_words


# A function to get list of words from given file:
def get_words(wordf, stopwords):
    words = []

    # Extract words to plot from file
    for word in open(wordf, "r", encoding="utf-8").read().split("\n"):
        words.append(word)

    # Check for a multi-word expression:
    for word in words:
        if "_" not in word:
            # This means it's a single word, so we get only its vector: so we skip this
            print("Single word:", word)
        else:
            # This means the expression is not a single word
            # We have to deal with this carefully:
            # First we split it to separate words, then remove stopwords:
            expression = word.split("_")
            expression_chunks = remove_stopwords(expression, stopwords)
            print("Found a stopword: {}, un-stopworded it:{}".format(word, ' '.join(expression_chunks)))

    return words


if __name__ == '__main__':
    # Calculating the time it took to process everything:
    start_time = time.time()
    print("Programm started.")

    print("Loading stopwords:")
    stopwords = load_stopwords(stopwords_file)

    print("Loading answer words from given file:")
    words = get_words(wordf, stopwords)
    
    print("Programm finished.")
    finish_time = time.time()
    exec_time = finish_time - start_time
    print("It took {} seconds to execute this.".format(round(exec_time)))

