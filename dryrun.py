#!/usr/bin/python
# import functions
import numpy as np
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
# import scripts
import antibody_helper_functions as fxns
# download data
# file_names gives which files are in the desired directory, given by PATH
PATH = "../../antibodies/isobel_data/2019_09_24_data_pipeline/"
file_names = []
for root, dirs, files in os.walk(PATH):
    for filename in files:
        file_names.append(filename)
# file_names_no_csv drops the .csv for ease of reading
file_names_no_csv = list(map(fxns.kill_csv, file_names))
# data_dict is a dictionary of dataframes for the different samples
data_dict = {}
for i in range(len(file_names)):
    data_dict[file_names_no_csv[i]] = pd.read_csv(PATH + file_names[i])
# combine all the data
raw_data_df = pd.concat(data_dict)
# drop everything not HIV 1
# should stop doing this in the future
hiv1_only_df = raw_data_df.loc[raw_data_df['species'] == 'Human immunodeficiency virus 1']
# recreate the antibody similarity matrix for s.p. 289
peptides_289 = hiv1_only_df.loc[hiv1_only_df['start'] == 289]['peptide_sequence'].unique()
# make small to make easier
peptides_289 = peptides_289[0:5]
# create similarity matrix
# score is given by alignments divided by length, so gives an output between 0 and 1
# definitely can change how the scoring is accomplished
print("made it to before constructing similarity matrix")
outer_lst = []
for p1 in peptides_289:
    inner_lst = []
    for p2 in peptides_289:
        alignments = pairwise2.align.globalxx(p1, p2)
        inner_lst.append(alignments[0][2] / float(alignments[0][4] - alignments[0][3]))
    outer_lst.append(inner_lst)
corr_mat = np.array(outer_lst)
corr_mat_df = pd.DataFrame(corr_mat, index = peptides_289)
print("made it to after constructing similarity matrix")
# make the cluster map
# sim_mat = sns.clustermap(corr_mat_df, cmap = 'jet', vmin = -1, vmax = 1, cbar_kws = {'label' : 'peptide sequence similarity'})
sim_mat = sns.clustermap(corr_mat_df)
print("made it to after the clustermap was called")
# how we may want to reorder peptides_289 so as to correspond to the clustering shown above
reorder_inds_peps_289 = sim_mat.dendrogram_col.reordered_ind
reordered_peps_289 = [peptides_289[x] for x in reorder_inds_peps_289]
reordered_corr_mat = [[corr_mat[x][y] for y in reorder_inds_peps_289] for x in reorder_inds_peps_289]
reordered_corr_mat_df = pd.DataFrame(reordered_corr_mat, index = reordered_peps_289)
print("made it to right before writing the file")
reordered_corr_mat_df.to_csv("pep_289_sim_mat.csv")