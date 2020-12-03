"""
ClusterDataSplit: companion code for the benchmarking study reported in the paper:

    ClusterDataSplit: Exploring Challenging Clustering-Based Data Splits for Model Performance Evaluation by Hanna Wecker, Annemarie Friedrich and Heike Adel. In Proceedings of Evaluation and Comparison of NLP Systems (Eval4NLP).

Copyright (c) 2019 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from sklearn.decomposition import PCA
from same_distribution_kmeans_updating_swaps import *
import numpy as np
import random
from sklearn import metrics
from IPython.display import display, Markdown
import os
import matplotlib.ticker as mtick
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.ticker import MaxNLocator, FuncFormatter
import collections
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict


"""
This notebook contains the functions called by the ClusterDataSplit Notebooks. 
For example, these functions generate plots or call the algorithms for cluster creation.

Author: Hanna Wecker
Author: Annemarie Friedrich
"""


##### FUNCTIONS USED FOR NOTEBOOK 1 - GENERAL OVERVIEW #####

def get_max_of_bin(value, bins):
    max_val = 0
    for bin_max in bins:
        max_val = bin_max
        if value < bin_max:
            break
    return max_val


def generate_overview_stats(text_data, labels, num_bins=10):
    """
    Generates number_ids, length_text, unique_labels and label_text_length from text and label information.
    :param text_data: list of strings. Contains text data for all examples.
    :param labels: 2-dimensional array. Contains gold labels for each example.
    :param num?bins: number of buckets/bins to group input text lengths
    :return:    number_ids: integer. Number of examples.
                length_text: 1-dim array. Number of tokens for each example.
                unique_labels. tuple. First array describes labels occuring, second array describes number of occurence
                    of the different labels.
                label_text_length: pandas dataframe. Columns "label" and "length_text" (measured in tokens).
    """
    number_ids = len(text_data)
    length_text = [len(string.split()) for string in text_data]
    # prepare bins
    max_text_length = max(length_text)
    bin_size = int(max_text_length/num_bins)
    bins = np.array([(x+1)*bin_size for x in range(num_bins)])
    length_text_binned = [get_max_of_bin(v, bins) for v in length_text]
    
    unique_labels = np.unique(labels, return_counts=True)
    idx2label = {i : l for i, l in enumerate(unique_labels[0])}
    label2idx = {l : i for i, l in enumerate(unique_labels[0])}
    
    # with numeric labels (indices)
    numeric_labels = [label2idx[l] for l in labels[0]]
    label_text_length = pd.DataFrame((numeric_labels), columns=['label'])
    label_text_length['length_text'] = length_text
    label_text_length['length_text_binned'] = length_text_binned
                                             
    return number_ids, length_text, unique_labels, label_text_length, idx2label, label2idx


def print_label_distribution(unique_labels, label2idx):
    """
    Displays the relative amount of examples with the different labels.
    :param unique_labels: tuple. First array describes labels occuring, second array describes number of occurence
                    of the different labels.
    :return: None.
    """
    for label in unique_labels[0]:
        print("Label {0:} {1:>10}: {2:.2%}".format(unique_labels[0][label2idx[label]], label, unique_labels[1][label2idx[label]] / sum(unique_labels[1])))



def plot_sentence_length(length_text):
    """
    Displays a histogram for text length measured by tokens.
    :param length_text: 1-dim array. Number of tokens for each example.
    :return: None.
    """
    f, ax = plt.subplots()
    plt.hist(length_text, bins=len(np.unique(length_text)), rwidth=0.8)
    plt.xlabel("Number of Tokens in Text")
    plt.ylabel("Number of Occurence")
    plt.title("Histogram Text Length")
    ax.set_xlim(xmin=0)
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()


def plot_label_distribution(unique_labels, label2idx=None):
    """
    Displays the relative amount of examples per label.
    :param unique_labels: tuple. First array describes labels occuring, second array describes number of occurence
                    of the different labels.
    :return: None.
    """
    ax = plt.figure(figsize=(7.5, 5)).gca()
    labels_for_legend = unique_labels[0]
    plt.bar(labels_for_legend, height=unique_labels[1] / sum(unique_labels[1]))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xlabel('Label')
    plt.ylabel('% of Examples')
    plt.title('Relative Amount of Examples per Label')


def generate_tokens_and_frequencies(text_data, rank):
    """
    Extracts the most <rank> most frequent tokens with their respective frequencies.
    :param text_data: list of strings. Contains text data for all examples.
    :param rank: integer. Number of tokens and their respective frequencies to be stored.
    :return:    tokens: tuple. Contains <rank> most frequent tokens sorted by frequency.
                frequencies: tuple. number of (absolute) occurence for <rank> most frequent tokens.
                full_text_string: string. Contains the text data from all examples merged in one string.
    """
    full_text_string = ' '.join(text_data)
    token_split = full_text_string.split()
    token_counter = collections.Counter(token_split)
    most_frequent = token_counter.most_common(rank)
    tokens, frequencies = zip(*most_frequent)

    return tokens, frequencies, full_text_string


def plot_frequent_tokens(tokens, frequencies):
    """
    plots the most frequent tokens.
    :param tokens: tuple. Contains most frequent tokens sorted by frequency.
    :param frequencies: tuple. number of (absolute) occurence for most frequent tokens.
    :return: None.
    """
    f, ax = plt.subplots()
    plt.bar(tokens, height=frequencies)
    plt.xticks(rotation=90)
    plt.title('Most Frequent Tokens')
    plt.xlabel("Tokens")
    plt.ylabel("Number of occurence")
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()


def generate_wordcloud(full_text_string, number_of_words):
    """
    Generates a wordcloud with the <number of words> most frequent words in the text.
    :param full_text_string: string. Contains the text data from all examples merged in one string.
    :param number_of_words: integer. Number of words to be displayed.
    :return: None.
    """
    cloud = WordCloud(background_color="white", max_words=number_of_words).generate(full_text_string)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def plot_tokens_by_label(text_data, unique_labels, label2idx, labels, rank):
    """
    Creates a separate plot with most frequent tokens for each label.
    :param text_data: list of strings. Contains text data for all examples.
    :param rank: Number of tokens to be displayed.
    :param unique_labels: tuple. First array describes labels occuring, second array describes number of occurence
                    of the different labels.
    :param labels: 2-dimensional array. Contains gold labels for each example.
    :return: None.
    """
    for label in unique_labels[0]:
        # extract example texts for the respective label.
        text_array = np.array(text_data)
        label_text = text_array[labels[0] == label]

        # extract most frequent tokens and frequencies for the respective label
        tokens, frequencies, _ = generate_tokens_and_frequencies(label_text, rank)

        # plot
        f, ax = plt.subplots()
        plt.bar(tokens, height=frequencies)
        plt.title('Most Frequent Tokens Label ' + str(label))
        plt.xticks(rotation=90)
        plt.xlabel("Tokens")
        plt.ylabel("Number of occurence")
        ax.get_yaxis().set_major_formatter(
            FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.show()
        
        # print word cloud, too
        print("Word cloud for label", str(label))
        generate_wordcloud(" ".join(label_text), number_of_words = 30)


def plot_labels_by_text_length(label_text_length, idx2label):
    """
    Visualize label distribution for varying text lengths.
    :param label_text_length: pandas dataframe. Columns "label" and "length_text" (measured in tokens).
    :return: None.
    """   

    # extract relative label distribution for each text length
    label_distribution_text = label_text_length.groupby(['length_text', 'label']).size()
    label_distribution_text = label_distribution_text.reset_index()
    label_distribution_text.columns = ['length_text', 'label', 'absolute']
    sum_labels_text = pd.DataFrame(label_distribution_text.groupby(['length_text']).sum())
    sum_labels_text.columns = ['label', 'absolute_sum']
    label_distribution_text = pd.merge(label_distribution_text, sum_labels_text, on='length_text')
    del label_distribution_text['label_y']
    label_distribution_text['relative'] = label_distribution_text['absolute'] / label_distribution_text['absolute_sum']
    
    # change to text labels for display
    label_distribution_text["label_x"] = label_distribution_text["label_x"].map(idx2label)

    # plot
    f, ax = plt.subplots()
    distribution_plot_text = sns.barplot(x="length_text", hue="label_x", y="relative", data=label_distribution_text)
    distribution_plot_text.set_title("Distribution of labels over text Length")
    distribution_plot_text.set_ylabel("Percentage for respective text length")
    distribution_plot_text.set_xlabel('Text Length (measured in tokens)')
    ax.legend(title='Label')
    for tick in distribution_plot_text.get_xticklabels():
        tick.set_rotation(90)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.show()


def plot_text_length_by_labels(unique_labels, label_text_length, label2idx):
    """
    Create box plot describing text length distribution per label.
    :param unique_labels: tuple. First array describes labels occuring, second array describes number of occurence
                    of the different labels.
    :param label_text_length: pandas dataframe. Columns "label" and "length_text" (measured in tokens).
    :return: None.
    """
    # create lists with the text length values for each of the labels in the dataset
    text_lengths_by_label = list([None] * len(unique_labels[0]))
    index = 0
    for label in unique_labels[0]:
        mask = (label_text_length['label'] == label2idx[label])
        one_label_data = label_text_length[mask]
        text_lengths_by_label[index] = np.array(one_label_data['length_text'])
        index += 1

    # plot as boxplot
    f, ax = plt.subplots()
    plt.boxplot(text_lengths_by_label, patch_artist=True)
    ax.set_xticklabels(unique_labels[0])
    plt.xlabel('Labels')
    plt.ylabel('Text length (measured in tokens)')
    plt.title('Boxplot Sentence Lengths by Labels')


##### FUNCTIONS USED FOR NOTEBOOK 2 - CREATING STRATEGIC DATA SPLITS #####

def check_options(sentence_embeddings, clustering_algorithm):
    """
    checks whether valid options for the variables "sentence_embeddings" and "clustering_algorithm" were selected.
    :param sentence_embeddings: string. Option for generating the sentence embedding.
    :param clustering_algorithm: string. Option for clustering the data.
    :return: None.
    """
    if sentence_embeddings not in ["Word2Vec", "customized"]:
        print('Option for sentence_embeddings unknown. Please choose either "Word2Vec" or "customized"')

    if clustering_algorithm not in ["distribution_sensitive_kmeans", "same_size_kmeans", "kmeans", "randomized"]:
        print(
            'Option for clustering_algorithm unknown. Please choose either "distribution_sensitive_kmeans", '
            '"same_size_kmeans", "kmeans" or "randomized"')


def create_sentence_vectors(text_data, model):
    """
    Creates sentence vectors by averaging the word vectors for each sentence.
    :param text_data: list. preprocessed input data, contains text string for each example.
    :param model: (gensim) model. Word2vec model that should be used for creating word vectors.
    :return:    sentence vectors: dictionary. Key: ID for each example. Value: example vector representation.
                exceptions. Integer. Number of exceptions. An exception means the model did not know all
                words in a sentence and thus had to leave out words in order to transform a sentence in a vector
                representation.
    """
    sentence_vectors = {}
    exceptions = 0
    
    vector_size = len(model[list(model.vocab.keys())[0]])  # retrieve length of vector

    for instance, text in enumerate(text_data):
        # returns dictionary with (average) sentence vectors. Key: ID in text_dataset
        try:
            vectors = [model[x] for x in text.split(' ')]
            sentence_vector = sum(vectors) / len(vectors)
        except:
            # exception if model does not know all words in a sentence
            sentence_vector = []
            vectors = []
            for word in text.split(' '):
                try:
                    # try to leave out the words the model doesn't know and obtain a sentence vector by averaging over
                    # the rest of the words
                    vectors.append(model[word])
                except:
                    pass
            try:
                sentence_vector = sum(vectors) / len(vectors)
            except:
                # if it is not possible to use any word of the sentence for creating the sentence vector, sentence
                # vector is left empty, using zeros here
                sentence_vector = [0 for i in range(vector_size)]
            exceptions += 1

        key = instance
        sentence_vectors[key] = sentence_vector

    return sentence_vectors, exceptions


def perform_pca(vectors, n_components):
    """
    reduces the number of dimensions by principal component analysis.
    :param vectors: 2-dim array. Vectors for which dimensionality reduction is desired. need to be centered and scaled!
    :param n_components: integer. Number of dimensions after reduction.
    :return:    (1) sentence_vectors: 2-dim array. vectors with reduced dimensionality
                (2) explained_variance_ratio. 1-dim array. percentage of variance in original data that is explained by
                the selected principal components
                (3) explained_variance: 1-dim array. absolute size of eigenvalues belonging to the selected principal
                components.
    """
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_
    reduced_vectors = pca.fit_transform(vectors)

    return reduced_vectors, explained_variance_ratio, explained_variance

def generate_default_label_distribution(gold_labels, num_clusters):
    """
    implements the same distribution that is present in the full data set for every cluster
    :param text_input_data: pandas dataframe. Contains label information in second column.
    :param num_clusters: integer. Number of clusters which should be formed in the data.
    :return: max_ids_labels. 2-dim numpy array. Contains information on how many examples for each label should be assigned
                to the different clusters (minimum and maximum number of instances per cluster)
    """
    labels = gold_labels
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label2idx = {}
    
    for i, l in enumerate(unique_labels):
        label2idx[l] = i
        
    for label in unique_labels:

        ids_per_cluster = (label_counts[label2idx[label]] / num_clusters)
        ceils = int(round(ids_per_cluster % 1, 2) * num_clusters)
        floors = num_clusters - ceils

        max_ids_label = np.array([[int(np.ceil(ids_per_cluster))] * ceils + [int(np.floor(ids_per_cluster))] * floors])

        if label2idx[label] == 0:
            max_ids_labels = max_ids_label

        elif label2idx[label] > 0:
            max_ids_labels = np.concatenate((max_ids_labels, max_ids_label))

    return max_ids_labels


def perform_kmeans(vectors, num_clusters):
    """
    Performs regular K-Means clustering
    :param vectors: 2-dim array. sentence vectors, centered and scaled.
    :param num_clusters: integer. Number of clusters which should be explored.
    :return:    (1) clustering_labels. 1-dim array. Cluster IDs assigned to the examples.
                (2) centroids. 2-dim array. Centroids associated to the clusters.
    """

    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=110)
    kmeans.fit(vectors)

    clustering_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return clustering_labels, centroids


def perform_kmeans_size(vectors, num_clusters, n_jobs, max_ids_labels=None):
    """
    Performs same_size K-Means algorithm by calling same size K-Means algorithm and assigning the same (fake) labek
    to all datapoints.
    :param vectors: 2-dim array. sentence vectors, centered and scaled.
    :param num_clusters: integer. Number of clusters which should be explored.
    :param n_jobs: integer. number of workers for parallelization.
    :param max_ids_labels: list. maximum number of ids per label in each cluster.
                           example: [np.array([1,1,1]), np.array([2,2,2])] means that from the first label,
                           there should be one example in each of the three clusters, from the second label there
                           should be three examples in each of the three clusters. Attention: the number of examples
                           assigned to the different clusters must exactly match the number of examples in the whole
                           dataset.
    :return:    (1) clustering_labels. 1-dim array. Cluster IDs assigned to the examples.
                (2) centroids. 2-dim array. Centroids associated to the clusters.
    """

    # create 'fake' gold labels: Same Size K-Means algorithm is executed when calling Size and Distribution Sensitive
    # algorithm and all points have the same label


    kmeans_size_distribution = KMeansDistribution(n_clusters=num_clusters, random_state=10,
                                                  max_ids_labels=max_ids_labels)

    gold_labels = np.array([np.repeat(0, len(vectors))])

    kmeans_size_distribution.fit(vectors, gold_labels, n_jobs)

    clustering_labels = kmeans_size_distribution.clustering_labels_
    centroids = kmeans_size_distribution.cluster_centers_

    return clustering_labels, centroids


def perform_kmeans_size_distribution(vectors, num_clusters, gold_labels, n_jobs, max_ids_labels=None):
    """
    Performs our size and distribution sensitive K-Means algorithm.
    :param vectors: 2-dim array. sentence vectors, centered and scaled.
    :param num_clusters: integer. Number of clusters to be created.
    :param gold_labels: 2-dim array. gold labels of the examples.
    :param n_jobs: integer. number of workers for parallelization.
    :param max_ids_labels: list. maximum number of ids per label in each cluster.
                           example: [np.array([1,1,1]), np.array([2,2,2])] means that from the first label,
                           there should be one example in each of the three clusters, from the second label there
                           should be three examples in each of the three clusters. Attention: the number of examples
                           assigned to the different clusters must exactly match the number of examples in the whole
                           dataset.
    :return:    (1) clustering_labels. 1-dim array. Cluster IDs assigned to the examples.
                (2) centroids. 2-dim array. Centroids associated to the clusters.
    """

    kmeans_size_distribution = KMeansDistribution(n_clusters=num_clusters, random_state=10,
                                                 max_ids_labels=max_ids_labels)
    
    # map gold labels to integers
    label2idx = {}
    for i, l in enumerate(np.unique(gold_labels)):
        label2idx[l] = i
        
    gold_labels = [[label2idx[l] for l in gold_labels[0]]]
    
    kmeans_size_distribution.fit(vectors, gold_labels, n_jobs)

    clustering_labels = kmeans_size_distribution.clustering_labels_
    centroids = kmeans_size_distribution.cluster_centers_

    return clustering_labels, centroids


def create_random_datasplits(vectors, num_folds):
    """
    Examples are assigned randomly to the different data folds.
    :param vectors: 2-dim array. sentence vectors.
    :param num_folds: integer. Number of data folds to be produced.
    :return:    fold_labels: 1-dim array. Data folds the examples belong to.
    """
    # splits should have same size --> determine how many ids are in data set
    num_ids = len(vectors)
    ids_per_fold = num_ids/num_folds
    fold_ids = list(range(0, num_folds))
    fold_labels = np.repeat(fold_ids, np.ceil(ids_per_fold))
    random.shuffle(fold_labels)
    fold_labels = fold_labels[0: num_ids]

    return fold_labels

def plot_clustering_output(n_components, sentence_vectors, labels_clustering, point_size = 5):
    """
    Create scatterplot with examples coloured according to cluster or fold assigned.
    :param n_components: integer. Number of dimensions the sentence vectors were reduced to in the PCA step.
    :param sentence_vectors: 2-dim array. sentence vectors.
    :param labels_clustering: 1-dim array. Cluster IDs/ fold IDs assigned to the examples.
    :return: None.
    """
    # reduce dimensionality of data to two dimensions (if note done already)
    if n_components > 2:
        sentence_vectors, explained_variance_ratio, explained_variance = perform_pca(sentence_vectors, 2)

    # find out which labels were assigned in clustering step
    # (for example, for the K-Means algorithm it is possible that clusters stay empty)
    cluster_ids = np.unique(labels_clustering)

    # create plot
    f, ax = plt.subplots()
    for i in cluster_ids:
        indexes = (labels_clustering == i)
        plot_vectors = sentence_vectors[indexes]
        # if one wants to save the plot later and the plot contains many points, set rasterized = True
        plt.scatter(plot_vectors.T[0], plot_vectors.T[1], label=i, s=point_size, rasterized=False)

    plt.title("Cluster Visualization PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.rcParams.update({'font.size': 64})
    # adjust legend
    lgnd = plt.legend(title="Cluster", loc='center left', bbox_to_anchor=(1, 0.5))
    for handle in lgnd.legendHandles:
        handle.set_sizes([20])
    #plt.savefig("kmeans_pca.pdf", bbox_inches = 'tight', dpi = 300)
    plt.show()


def save_output_files(labels_clustering, n_components, gold_labels, sentence_vectors, clustering_algorithm, path):
    """
    Saves output files "analysis" and "datasplits" for the clustering_algorithm.
    :param labels_clustering: 1-dim array. Cluster IDs/ fold IDs assigned to the examples.
    :param n_components: integer. Number of dimensions the sentence vectors were reduced to in the PCA step.
    :param gold_labels: 2-dim array. gold labels of the examples.
    :param sentence_vectors: 2-dim array. sentence vectors.
    :param path: string. Path where output files should be stored.
    :return: None.
    """
    # storing clustering result in .tsv file with ID and cluster_label information
    ids = np.array(range(len(labels_clustering)))
    clustering_output = np.array([ids, labels_clustering.astype(int)])
    clustering_output = pd.DataFrame(clustering_output.T, columns=["ids", "clustering_labels"])
    clustering_output.to_csv(path_or_buf=os.path.join(path, "datasplits_" + clustering_algorithm + '.tsv'), sep="\t", index=False)

    # assure that sentence vectors only have two dimensions (otherwise, it is impossible to plot the examples in
    # two dimensions in the analysis notebook).
    if n_components > 2:
        sentence_vectors, explained_variance_ratio, explained_variance = perform_pca(sentence_vectors, 2)

    # store additional output file with vector information as input for performance analysis file
    labels_and_vectors = np.append(gold_labels.T, sentence_vectors, axis=1)
    analysis_input = clustering_output.join(pd.DataFrame(labels_and_vectors))
    analysis_input.columns = ["ids", "clustering_labels", "gold_labels", "PC1", "PC2"]
    analysis_input.to_csv(path_or_buf = os.path.join(path, "analysis_" + clustering_algorithm + ".tsv"), sep="\t", index=False)


##### FUNCTIONS FOR NOTEBOOK 3: ANALYZING PERFORMANCE WITH REGARD TO DATA SPLIT CHARACTERISTICS #####


def process_inputs(input_data, data_split_info):
    """
    Adds text length information to data_split_info and extracts the cluster IDs assigned.
    :param input_data: pandas dataframe. Columns "labels", "text_input_data".
    :param data_split_info: pandas dataframe. Columns: "ids", "clustering_labels", "gold_labels", "PC1", "PC2".
    :return:    data_split_info. pandas dataframe. same as input "data_split_info" with additional column "text_length"
                cluster_ids. 1-dim array. Cluster IDs present in the data.
    """
    text_data = list(input_data['text_input_data'])
    text_length_info = create_text_length_information(text_data)
    data_split_info['text_length'] = text_length_info

    cluster_ids = np.unique(data_split_info['clustering_labels'])

    return data_split_info, cluster_ids


def create_text_length_information(text_data):
    """
    Counts the tokens in the text strings.
    :param text_data: list. Contains text string for each example.
    :return:    text_length_info: list. Token text lengths.
    """
    text_length_info = []
    for text in text_data:
        text_length = len([x for x in text.split(' ')])
        text_length_info.append(text_length)

    return text_length_info


def calculate_f1_scores(data_split_info, cluster_ids):
    """
    calculates F1-scores for each cluster predicted, average F1 scores and standard deviations.
    :param data_split_info: pandas dataframe. contains i.a. id and clustering label information.
    :param cluster_ids: 1-dim array. cluster IDs present in partition.
    :param predictions: dictionary. keys: cluster ID (as integer). values: predictions obtained for the examples in
                            cluster (sorted by example IDs).
    :return:performance. pandas dataframe. Contains F1-scores for each cluster, average F1-scores and standard
                                            deviation.
    """
    
    # binary or multi-class?
    avg = "macro"
    label_set = np.unique(data_split_info['gold_labels'])
    if len(label_set) == 2:
        avg = "binary"
    
    # generate metrics for CV setting
    f1_scores = []
    
    for i in cluster_ids:
        indexes = (data_split_info['clustering_labels'] == i)
        # extract gold labels
        fold_gold_labels = data_split_info['gold_labels'][indexes]
        fold_predictions = data_split_info['predictions'][indexes]
        fold_f1 = metrics.f1_score(fold_gold_labels, fold_predictions, average=avg)
        f1_scores.append(fold_f1)

    # mean and variances
    f1_scores.append(np.mean(f1_scores))
    f1_scores.append(np.std(f1_scores))

    # store scores in dataframe
    performance = pd.DataFrame(f1_scores).transpose()
    key_strings = [str(key) for key in [*cluster_ids]]
    performance.columns = key_strings + ['mean', 'std']

    return performance

def display_f1_scores(performance, axis_value, type):
    """
    Plots a matplotlib table showing the F1 calculations.
    :param performance: pandas dataframe. Contains F1-scores for each cluster, average F1-scores and standard deviation.
    :param axis_value: axes subplot object. Describes where subplot should be placed.
    :param type: string. Either "Benchmark" or "Experiment" (depending on the origin of the data for the calculations).
    :return: None.
    """

    axis_value.axis('off')
    axis_value.axis('tight')
    axis_value.set_title(type, size=20)
    table = axis_value.table(cellText=np.around(performance.values, 2), colLabels=performance.columns, loc='center',
                             bbox=[0, 0.3, 1, 0.4], rowLabels=['F1-Scores'])
    table.set_fontsize(50)


def plot_clustering(cluster_ids, data_split_info, axis_value, point_size = 5):
    """
    Displays a scatter plot with points and the assigned cluster IDs.
    :param cluster_ids: 1-dim array. Cluster IDs present in the data.
    :param data_split_info: pandas dataframe. Contains i.a. example id and assigned cluster information.
    :param axis_value: axes subplot object. Describes where subplot should be placed.
    :return:
    """
    for i in cluster_ids:
        indexes = (data_split_info['clustering_labels'] == i)
        plot_vectors = data_split_info[indexes]
        # if one wants to save the plot later and the plot contains many points, set rasterized = True
        axis_value.scatter(plot_vectors['PC1'], plot_vectors['PC2'], label=i, s=point_size, rasterized=False)

    axis_value.set_title("Cluster Visualization PCA")
    axis_value.set_xlabel("PC1")
    axis_value.set_ylabel("PC2")
    # adjust legend
    lgnd = axis_value.legend(title="Cluster", loc='center left', bbox_to_anchor=(1, 0.5))
    for handle in lgnd.legendHandles:
        handle.set_sizes([20])

def plot_relative_size(data_split_info, axis_value):
    """
    visualizes relative size of each cluster.
    :param data_split_info: pandas dataframe. Contains i.a. example id and assigned cluster information.
    :param axis_value: axes subplot object. Describes where subplot should be placed.
    :return: none.
    """
    # calculate and store relative sizes
    total_datapoints = len(data_split_info)
    cluster_size = pd.DataFrame()
    cluster_size['absolute_size'] = data_split_info[['clustering_labels']].groupby(['clustering_labels']).size()
    cluster_size['relative_size'] = cluster_size['absolute_size'] / total_datapoints
    cluster_size['cluster_id'] = cluster_size.index

    # visualize results of calculations
    axis_value.bar(list(cluster_size['cluster_id']), list(cluster_size['relative_size']), width=0.7)
    axis_value.grid(axis='x')
    axis_value.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axis_value.set_title('Relative Size of Clusters')
    axis_value.set_xlabel('Cluster')
    axis_value.set_ylabel('Relative Amount of Datapoints')
    axis_value.set_xticks(cluster_size['cluster_id'])


def plot_label_distribution_analysis(data_split_info, axis_value):
    """
    shows relative label distribution in each of the clusters.
    :param data_split_info: pandas dataframe. Contains i.a. example id and assigned cluster information.
    :param axis_value: axes subplot object. Describes where subplot should be placed.
    :return: None.
    """    
    # determine mean label distribution in whole dataset
    mean_labels = pd.DataFrame(data_split_info[['gold_labels']].groupby(['gold_labels']).size() / len(data_split_info))

    # determine relative label distribution in each cluster
    label_distribution = data_split_info[['clustering_labels', 'gold_labels']].groupby(
        ['clustering_labels', 'gold_labels']).size()
    label_distribution = label_distribution.reset_index()
    label_distribution.columns = ['cluster_id', 'label', 'absolute']
    sum_absolute_labels = label_distribution.groupby(['cluster_id']).sum()
    sum_absolute_labels.columns = ['label', 'absolute_sum']
    label_distribution = pd.merge(label_distribution, sum_absolute_labels, on='cluster_id')
    del label_distribution['label_y']
    label_distribution['relative'] = label_distribution['absolute'] / label_distribution['absolute_sum']
    label_distribution['cluster_id'] = label_distribution['cluster_id'].astype(int)
    label_distribution['label_x'] = label_distribution['label_x'].astype(int)

    # plot results
    distribution_plot = sns.barplot(x="cluster_id", hue="label_x", y="relative", data=label_distribution,
                                    palette={0: "#4878CF", 1: "#6ACC65"}, ax=axis_value)
    distribution_plot.axhline(mean_labels.iloc[0, 0], color="#4878CF")
    distribution_plot.axhline(mean_labels.iloc[1, 0], color="#6ACC65")
    distribution_plot.set_title("Label Distributions for Clusters")
    distribution_plot.set_xlabel("Cluster")
    distribution_plot.set_ylabel("Relative Amount of Datapoints")
    distribution_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Labels")
    distribution_plot.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

def plot_sentence_length_distribution(data_split_info, axis_value):
    """
    displays mean sentence length in each cluster.
    :param data_split_info: pandas dataframe. Contains i.a. example id and assigned cluster information.
    :param axis_value: axes subplot object. Describes where subplot should be placed.
    :return: None.
    """
    # determine mean sentence length
    mean_sentence_length = data_split_info['text_length'].mean()

    # plot sentence lengths per cluster and mean sentence length
    sentence_length_plot = data_split_info[['clustering_labels', 'text_length']].groupby(
        ['clustering_labels']).mean().plot(kind='bar', ax=axis_value)
    sentence_length_plot.axhline(mean_sentence_length)
    sentence_length_plot.set_title("Mean Sentence Length for Clusters")
    sentence_length_plot.legend(title="Text Length", loc='center left', bbox_to_anchor=(1, 0.5))
    sentence_length_plot.set_ylabel("Text Length")
    sentence_length_plot.set_xlabel('Cluster')
    sentence_length_plot.set_xticklabels(sentence_length_plot.get_xticklabels(), rotation=0)
    
def read_predictions(pred_file, data_split_info):
    """ Reads predictions from data file and adds them to data_split_info.
    Careful: assumes instance ids are integers corresponding to the rows in that dataframe.
    """
    predictions = defaultdict(list)  # cluster id: predictions for instances of the fold sorted by example id
    gold = defaultdict(list)
    data_splits = {row[1]["ids"] : row[1]["clustering_labels"] for row in data_split_info.iterrows()}
    # data_splits: id to fold
    with open(pred_file) as f:
        next(f) # skip header
        for line in f:
            data = line.strip().split("\t")
            example, pred_label = int(data[0]), data[2]
            data_split_info.at[example, "predictions"] = pred_label

def generate_experiment_and_benchmark_plots(data_split_name, data_split_info, cluster_ids,
                                            benchmark_data_split_name, benchmark_data_split_info,
                                            benchmark_cluster_ids, binary_task=False):
    """
    Plots experiment and benchmark values.
    :param data_split_name: string. Type of data split (entered by user).
    :param data_split_info: pandas dataframe. contains i.a. id and clustering label information.
    :param cluster_ids: 1-dim array. cluster IDs present in partition.
    :param: benchmark_data_split_name: string. Type of benchmark data split (entered by user).
    :param benchmark_data_split_info: same data type as data_split_info, values for benchmark.
    :param benchmark_cluster_ids: same data type as cluster_ids, values for benchmark.
    :param binary_task: whether the classification task is binary or not (can display some additional plots in this case)
    :return: None.
    """
    if binary_task:
        f, axs = plt.subplots(5, 2, figsize=(20, 15))
    else:
        f, axs = plt.subplots(3, 2, figsize=(20, 15))

    for type in ['Experiment', 'Benchmark']:

        if type == 'Experiment':
            name = data_split_name
            data_split_info = data_split_info
            cluster_ids = cluster_ids
            cols = 0
        elif type == 'Benchmark':
            name = benchmark_data_split_name
            data_split_info = benchmark_data_split_info
            cluster_ids = benchmark_cluster_ids
            cols = 1

        # calculate and display performance
        performance = calculate_f1_scores(data_split_info, cluster_ids)
        display_f1_scores(performance, axis_value = axs[0, cols], type = name)

        # plot clustering
        plot_clustering(cluster_ids, data_split_info, axis_value = axs[1, cols], point_size = 2)

        # plot relative size
        plot_relative_size(data_split_info, axis_value = axs[2, cols])
        
        if binary_task:
            # These functions have been implemented only for the binary case to date.
            # Need to be adapted to multi-class scenario.
            
            # plot label distribution in each cluster
            plot_label_distribution_analysis(data_split_info, axis_value=axs[3, cols])

            # plot sentence length distribution
            plot_sentence_length_distribution(data_split_info, axis_value=axs[4, cols])


def generate_experiment_plots(data_split_name, data_split_info, cluster_ids, binary_task=False):
    """
    plots experiment results only.
    :param data_split_name: string. Type of data split (entered by user).
    :param data_split_info: pandas dataframe. contains i.a. id and clustering label information.
    :param cluster_ids: 1-dim array. cluster IDs present in partition.
    :param predictions: dictionary. keys: cluster ID (as integer). values: predictions obtained for the examples in
                            cluster (sorted by example IDs).
    :param binary_task: whether the classification task is binary or not (can display some additional plots in this case)
    :return: None.
    """

    if binary_task:
        f, axs = plt.subplots(5, 1, figsize=(20, 15))
    else:
        f, axs = plt.subplots(3, 1, figsize=(20, 15))

    # calculate and display performance
    performance = calculate_f1_scores(data_split_info, cluster_ids)
    display_f1_scores(performance, axis_value=axs[0], type=data_split_name)

    # plot clustering
    plot_clustering(cluster_ids, data_split_info, axis_value=axs[1], point_size = 2)

    # plot relative size
    plot_relative_size(data_split_info, axis_value=axs[2])
    
    if binary_task:
        # These functions have been implemented only for the binary case to date.
        # Need to be adapted to multi-class scenario.
        
        # plot label distribution in each cluster
        plot_label_distribution_analysis(data_split_info, axis_value=axs[3])

        # plot sentence length distribution
        plot_sentence_length_distribution(data_split_info, axis_value=axs[4])


def generate_plots(binary_task, data_split_name, data_split_info, cluster_ids, include_benchmark,
                   benchmark_data_split_name = None, benchmark_data_split_info=None,
                   benchmark_cluster_ids=None):
    """
    Visualizes either the experiment results only or the experiment and the benchmark results
    (depending if the inclusion of the benchmark was selected).
    :param data_split_name: string. Type of data split (entered by user).
    :param data_split_info: pandas dataframe. contains i.a. id and clustering label information.
    :param cluster_ids: 1-dim array. cluster IDs present in partition.
    :param predictions: dictionary. keys: cluster ID (as integer). values: predictions obtained for the examples in
                            cluster (sorted by example IDs).
    :param include_benchmark: bool. Flag indicating whether the benchmark values should be included in the
                                visualization.
    :param: benchmark_data_split_name: string. Type of benchmark data split (entered by user).
    :param benchmark_data_split_info: same data type as data_split_info, values for benchmark.
    :param benchmark_cluster_ids: same data type as cluster_ids, values for benchmark.
    :param benchmark_predictions: same data type as predictions, values for benchmark.
    :param binary_task: whether the classification task is binary or not (can display some additional plots in this case)
    :return: None.
    """
    if include_benchmark:
         generate_experiment_and_benchmark_plots(data_split_name, data_split_info, cluster_ids,                                              benchmark_data_split_name, benchmark_data_split_info,
                                                benchmark_cluster_ids, binary_task)
    else:
        generate_experiment_plots(data_split_name, data_split_info, cluster_ids, binary_task)

    plt.tight_layout()


