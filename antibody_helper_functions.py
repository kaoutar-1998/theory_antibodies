# helper functions for your analysis of data from Isobel Hawes, for Le Yan and the biohub
# import libraries
import numpy as np
import math
import scipy
# import pandas to work with data
import pandas as pd
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
# import itertools, reduce to examine intersection of four different sampling dates
from itertools import chain
from itertools import combinations
from itertools import product
from functools import reduce
# processing data
from sklearn.preprocessing import StandardScaler
# import Biopython
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.Alphabet import generic_nucleotide
from Bio.SeqUtils import GC
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
# File I/O
import os
# Helper functions
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
#normalize the rows of a matrix
# notice that this will NOT work for normalizing a single vector!!!!!!!!
def normalize_rows(arr):
    return np.array([x / np.linalg.norm(x) for x in arr])
# for given array (with rows normalized), given the cosine similarity matrix of the rows
def cos_sim_mat(arr):
    mat = []
    for x in arr:
        row = []
        for y in arr:
            row.append(np.dot(x,y))
        mat.append(row)
    return np.array(mat)
# gives average cosine similarity for a matrix showing cosine similarities between each pair of vector
# ignores diagonal of all ones
def avg_cos_sim_diag(arr):
    n = len(arr)
    return (arr.sum() - n) / (n ** 2 -n)
# gives average cosine similarity for a matrix showing cosine similarities between each pair of vector
# includes diagonal b/c this is for blocks where no vector is being compared with itself
def avg_cos_sim_no_diag(arr):
    r,c = arr.shape
    return (arr.sum()) / (r * c)
def mat_cuts(lst):
    return [0] + list(np.cumsum(lst))
# this function takes in cos_sim_mat, which is a matrix consisting of the cosine similarities betweeen different vectors, 
# and splits them according to the different patients (according to split_lst)
def cos_sim_block_mat(cos_sim_mat, split_lst):
    cuts = mat_cuts(split_lst)
    outer_lst = []
    for i in range(len(split_lst)):
        inner_lst = []
        for j in range(len(split_lst)):
            block_mat = cos_sim_mat[cuts[i]: cuts[i + 1], cuts[j]: cuts[j + 1]]
            if i == j:
                avg_cos_sim = avg_cos_sim_diag(block_mat)
            else:
                avg_cos_sim = avg_cos_sim_no_diag(block_mat)
            inner_lst.append(avg_cos_sim)
        outer_lst.append(inner_lst)
    return outer_lst
#remove .csv ending from file names
def kill_csv(name):
    return name[:-4]
#sanity check to perform after any operation on a dataframe
def sanity_check_df(df):
    print(df.shape)
    display(df.head())
    display(df.tail())
# gives the amount of peptide sequence p measured in sample s
def extract_from_hiv_by_sample(s,p):
    curr_df = hiv1_only_df.loc[s]
    arr = np.array(curr_df.loc[curr_df['peptide_sequence'] == p]['reads_per_100K_(rpK)'])
    if arr.size == 0:
        return 0
    else:
        return arr[0]
# natural logarithm with minimum value set to 0, as no peptide with rpK less than 1 is reported in the data
def soft_log(x):
    return np.log(x + 1)
# create dictionary where key is sample id, value are names of peptides measured in this sample with start position s
def date_peptide_s_dictionary(s):
    date_peptide_dictionary = {}
    for x in file_names_no_csv:
        curr_df = hiv1_only_df.loc[x]
        date_peptide_dictionary[x] = set(curr_df.loc[curr_df['start'] == s]['peptide'])
    return date_peptide_dictionary
# gives list of all peptides SEQUENCES in data with given starting position s
def peptides_s(s):
    return hiv1_only_df.loc[hiv1_only_df['start'] == s]['peptide_sequence'].unique()
# gives the similarity matrix between peptides sequences in given list pep_seq
# score is given by alignments divided by length, so gives an output between 0 and 1
# definitely can change how the scoring is accomplished
# should make formula clear and have examples
def pep_sim_mat(pep_lst):
    outer_lst = []
    for p1 in pep_lst:
        inner_lst = []
        for p2 in pep_lst:
            alignments = pairwise2.align.globalxx(p1, p2)
            inner_lst.append(alignments[0][2] / float(alignments[0][4] - alignments[0][3]))
        outer_lst.append(inner_lst)
    sim_mat_raw = np.array(outer_lst)
    sim_mat_df = pd.DataFrame(sim_mat_raw, index = pep_lst)
    sim_mat_clust = sns.clustermap(sim_mat_df, cmap = 'jet', vmin = -1, vmax = 1, cbar_kws = {'label' : 'peptide sequence similarity'})
    reorder_inds_peps = sim_mat_clust.dendrogram_col.reordered_ind
    reordered_peps = [pep_lst[x] for x in reorder_inds_peps]
    reordered_sim_mat = [[sim_mat_raw[x][y] for y in reorder_inds_peps] for x in reorder_inds_peps]
    reordered_sim_mat_df = pd.DataFrame(reordered_sim_mat, index = reordered_peps)
    reordered_sim_mat_df.columns = reordered_peps
    return reordered_sim_mat_df
# for given eigenvectors, principal component start and end range, data, and patient splits, gives the enrichments, the cosine similarities between samples, and the average cosine similarity within and between patients
# data goes in as a dataframe
def component_exploration(evecs, start, stop, data_df, patient_splts):
    # first we show the enrichment scores
    unnormalized_enrichment = np.matmul(np.array([x for x in data_df.values]), evecs[:,start:stop])
    normalized_enrichment = normalize_rows(unnormalized_enrichment)
    ax = plt.axes()
    y_axis_labels = file_names_no_csv # labels for y-axis
    ax.set_title("Enrichment scores of principal components " + str(start) + " to " + str(stop - 1) + " (log rpK)")
    sns.heatmap(normalized_enrichment, cmap = 'jet', yticklabels = y_axis_labels, ax = ax, cbar_kws = {'label' : 'enrichment score'})
    ax.set_ylim(len(data_df),0) # hard code b/c heatmap no longer works with matplotlib
    plt.yticks(rotation=0) 
    plt.xlabel('principal components')
    plt.show()
    plt.close()
    # then we show the cosine similarity between patients
    cos_sim_enriched = cos_similarity_patient_splits(normalized_enrichment, file_names_no_csv, patient_splts)
    # then we return the average similarity between patients
    # should we show stdv as well?
    return cos_sim_block_mat(cos_sim_enriched, patient_splts)
# gives the cosine similarity matrix, delineating the splits between patients
#sample_scores gives the vectors for each patient sample
#sample_names give the name of each patient sample
# title is the string that gives the title of each patient
def cos_similarity_patient_splits(sample_scores, sample_names, patient_splts, title):
    cos_sim = cos_sim_mat(normalize_rows(sample_scores))
    ax = plt.axes()
    y_axis_labels = sample_names # sample names
    x_axis_labels = sample_names
    ax.set_title("Cos. sim. between patients" + title)
#     mask = np.zeros_like(cos_sim)
#     mask[np.diag_indices_from(mask)] = True
#     sns.heatmap(cos_sim, mask = mask, cmap = 'jet', xticklabels = x_axis_labels, yticklabels = y_axis_labels, ax = ax, cbar_kws = {'label' : 'cosine similarity'})
    sns.heatmap(cos_sim, cmap = 'jet', vmin = 0, vmax = 1, xticklabels = x_axis_labels, yticklabels = y_axis_labels, ax = ax, cbar_kws = {'label' : 'cosine similarity'})
    ax.set_ylim(len(sample_scores),0)
    plt.yticks(rotation=0) 
    ax.hlines(mat_cuts(patient_splts), *ax.get_xlim()) #lines to indicate patient grouping
    ax.vlines(mat_cuts(patient_splts), *ax.get_ylim()) #lines to indicate patient grouping
    plt.show()
    plt.close()
    return cos_sim
# reports the mean and standard deviation of each row in a matrix
# meant to be used for cos_sim_block_mat
def mean_and_std_by_row(arr):
    mat = []
    for r in arr:
        m = np.mean(r)
        s = np.std(r)
        mat.append([m,s])
    return mat
# produces a nicely labeled heatmap of data in dataframe (df)
# provide strings for"
# the title of the plot (title)
# what the colormap menas (cbar)
# x-axis and y-axies label (xlabel and label)
def label_heatmap(df, title, cbar, xlabel, ylabel):
    ax = plt.axes()
#     y_axis_labels = file_names_no_csv
    ax.set_title(title)
    sns.heatmap(df, cmap = 'jet', ax = ax, cbar_kws = {'label' : cbar})
    ax.set_ylim(len(df),0) # hard code b/c heatmap no longer works with matplotlib
    plt.yticks(rotation=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()
    # gives heatmap of the average cosine similarity within and between patients
# takes in the title (a string)
# and the cos_sim_mat of all the patient samples with each other
def avg_cos_sim_heatmap(title, cos_sim_mat):
    ax = plt.axes()
    y_axis_labels = patient_lst
    x_axis_labels = patient_lst
    ax.set_title('Avg. cos. sim. between patients ' + title)
    sns.heatmap(cos_sim_block_mat(cos_sim_mat, split_sample_lst), vmin = 0, vmax = 1, cmap = 'jet', annot=True, ax = ax, yticklabels = y_axis_labels, xticklabels = x_axis_labels, cbar_kws = {'label' : "avg cos sim"})
    ax.set_ylim(len(patient_lst),0) # hard code b/c heatmap no longer works with matplotlib
    plt.yticks(rotation=0)
    plt.show()
    plt.close()
