<div id="top"></div>

<!-- PROJECT SHIELDS -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">K-menas Clustering of Survey Answers</h3>
  <p align="center">
    Answers of a survey in Spanish are categorized using word-embeddings, and categorized using k-means clustering. <br />
    This project also includes dealing with multi-word expressions, by removing stopwords, and obtaining their vector-averages. <br />
    This is an example for Spanish language but it can easily be adapted for any other languages.<br />
    The number of clusters are obtained by optimal curvage finding algorithm(elbow method).
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#how-it-works">How it works</a></li>
    <li><a href="#usage">How to use</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#Acknowledgments">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<div align="center">
<img src="https://github.com/elmurod1202/survey-clustering/blob/main/src/example-figure.png?raw=true" width = "700" Alt = "Spanish words scattered in 2D space">
</div>

This project was created with a purpose to serve people who are searching for a solution to group/categorize words or even multi-word expressions by their meaning. There are so many tools and services to run statistics and/or create diagrams of given data, but they mostly work for numbers, when it comes to deal with words or some texts, those tools seem less useful since they do not include any way to visualize them in 2D/3D space based on their usage/meaning. This repository somewhat helps to perform following operations on texts:


* Visualising texts:
  * Visualizing single words using word-embedding vectors of a language;
  * Visualizing multiple-word texts by obtaining average vecotrs of containing words (stopwords removed for better output quality);
* Finding the optimal number of groups/clusters/categories to split words/texts based on their meaning, using Within Cluster Sum of Squares(WCSS) to find a level-oof using elbow method;
* Grouping/Clustering texts using k-means clustering algorithm;
* Visualizing grouped texts by different colors, using patplotlib.


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

Programming language used:

* [Python](https://www.python.org/)

These are the major libraries used inside Python:

* [scikit-learn : A set of python modules for machine learning](https://scikit-learn.org/stable/)
* [gensim: Python framework for fast Vector Space Modelling](https://pypi.org/project/gensim/)
* [Matplotlib: Visualization with Python](https://matplotlib.org/)
* [kneed: Knee-point detection in Python](https://pypi.org/project/kneed/)
* [NumPy: The fundamental package for scientific computing with Python](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- How it works  -->
## How it works
* First of all, the code loads the list of words/texts from given file: *input/answers.txt* (it's called answers in this case, because it was an answers of a particular survey), and it obtains a vectors of those texts. An example diagram would look like this:
<div align="center">
<img src="https://github.com/elmurod1202/survey-clustering/blob/main/output/result_scatter_2d.png?raw=true" width = "500" Alt = "Spanish words scattered in 2D space">
</div>

* Then, the code obtains the optimal number of clusters for given texts to splin into, using an elbow-method. For our example it would look like this:
<div align="center">
<img src="https://github.com/elmurod1202/survey-clustering/blob/main/output/elbow_method_2d.png?raw=true" width = "500" Alt = "Elbow method example diagram">
</div>

* Lastly, the code categorizes the list of texts into groups by their meaning. The final result would look like this:
<div align="center">
<img src="https://github.com/elmurod1202/survey-clustering/blob/main/output/result_words_2d.png?raw=true" width = "500" Alt = "Clustered words in a scatter">
</div>


<p align="right">(<a href="#top">back to top</a>)</p>




<!-- How to use  -->
## Usage

To use this code you should have at least a small understanding of how to run a Python code, with Python installed machine. You should also install above-mentioned necessary framework/libraries into it.
There are two ways you can run this code: 
1. Either clone the repo by running the commend below, and run the survey-clustering.py:

   ```sh
   git clone https://github.com/elmurod1202/survey-clustering.git
   ```
2. Or just download only the *survey-clustering.py* (or *survey-clustering-minimum.py* if you want minimised working code without graphic visualisations) file and make some small changes like where to read the files from and where to store the results to. That's it.

***IMPORTANT:*** This code uses a Spanish word embeddings vector file that is not inluded here due to its big size. 
Please download the file into the src/ folder from the link: <a href="https://zenodo.org/record/3234051/files/embeddings-l-model.vec?download=1">Spanish word vectors (3.4 GB)</a>

### Adapting for other languages:
This code is ontended for Spanish, but it can be adapted to many other languages just by changing two files in the src/ folder: 
* src/embeddings-l-model.vec : Spanish word vectors file to a word-vector file of any language;
* src/spanish-stopwords.txt : Spanish stopwords file replaced by any toher language stopwords.


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU GENERAL PUBLIC LICENSE. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Big shoutouts to <a href = "https://www.linkedin.com/in/luissantalla/">Luis</a>  for bringing this problem to the table. 

We are grateful for these resources and tutorials for making this repository possible:

* [Tutorial on Visualising words](https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5)
* [Tutorial on K-means clustering](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203)
* [Tutorial on K-means clustering](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203)
* [Spanish pretrained word-embeddings](https://github.com/dccuchile/spanish-word-embeddings)
* [GitHub Readme template](https://github.com/othneildrew/Best-README-Template)
* [Visual Studio Code](https://code.visualstudio.com/)


<p align="right">(<a href="#top">back to top</a>)</p>

