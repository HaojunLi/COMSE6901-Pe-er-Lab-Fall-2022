# Fall 2022 Pagel's lambda
from os import name
from biom import load_table
from re import L
import numpy as np
import math
from numpy.core.fromnumeric import ptp
from numpy.linalg import inv, det, pinv, slogdet
import pandas as pd
from ete3 import Tree
from Bio import Phylo
from six import b
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import biom
# Brownian motion likelihood computation
# Brownian motion models can be completely described by two parameters. The first is the starting value of the population mean trait, z¯(0).
# This is the mean trait value that is seen in the ancestral population at the start of the simulation, before any trait change occurs.
# The second parameter of Brownian motion is the evolutionary rate parameter, σ^2. This parameter determines how fast traits will randomly walk through time.

# Under Brownian motion, changes in trait values over any interval of time are always drawn from a normal distribution
# with mean 0 and variance proportional to the product of the rate of evolution and the length of time (variance = σ^2t).
# x is an n x 1 vector of trait values for the n tip species in the tree

np.seterr(over='raise')

def Z0_delatSquare_basic_Unittest(val, text):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    if type(val) != type(randomMatrix):
        raise TypeError(f"{text} should be a numpy array")
    if len(val.shape) != 2:
        raise ValueError(f"{text} should be np.array([[value]])")
    if val.shape[0] != 1 or val.shape[1] != 1:
        raise ValueError(f"{text} should be np.array([[value]])")
    if type(val[0][0]) != type(randomNumpyInt) and type(val[0][0]) != type (randomNumpyFloat):
        raise TypeError(f"Value in {text} sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")

# x is an n x 1 vector of trait
def Basic_unittest(X,Z0, deltaSquare, C):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    # X OK
    if type(X) != type(randomMatrix):
        raise TypeError("X should be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("X should be an n x 1 vector")
    if X.shape[1] != 1:
        raise ValueError("X should be an n x 1 vector")
    for x in X:
        j = x[0]
        if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
            raise TypeError(f"Value in X sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
    # Z0 OK
    Z0_delatSquare_basic_Unittest(Z0, "Z0")
    
    # deltaSquare OK
    Z0_delatSquare_basic_Unittest(deltaSquare, "deltaSquare")
    if deltaSquare[0][0] < 0:
        raise ValueError("deltaSquare should be greater than or equal to 0")
        
    # C OK
    if type(C) != type(randomMatrix):
        raise TypeError("C should be a numpy array")
    if len(C.shape) != 2:
        raise ValueError("C should be an n x n symmetric matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C should be an n x n symmetric matrix")
    if not np.array_equal(C, C.T):
        raise ValueError("C should be an n x n symmetric matrix")
    for c in C:
        for j in c:
            if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
                raise TypeError(f"Value in C sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
    if X.shape[0] != C.shape[0]:
        raise ValueError("X should be an n x 1 vector and C should be an n x n symmetric matrix")
# Finished

def Basic_unittest_X_C(X, C):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    # X OK
    if type(X) != type(randomMatrix):
        raise TypeError("X should be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("X should be an n x 1 vector")
    if X.shape[1] != 1:
        raise ValueError("X should be an n x 1 vector")
    for x in X:
        j = x[0]
        if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
            raise TypeError(f"Value in X sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")        
    # C OK
    if type(C) != type(randomMatrix):
        raise TypeError("C should be a numpy array")
    if len(C.shape) != 2:
        raise ValueError("C should be an n x n symmetric matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C should be an n x n symmetric matrix")
    if not np.array_equal(C, C.T):
        raise ValueError("C should be an n x n symmetric matrix")
    for c in C:
        for j in c:
            if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
                raise TypeError(f"Value in C sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
    if X.shape[0] != C.shape[0]:
        raise ValueError("X should be an n x 1 vector and C should be an n x n symmetric matrix")
# Finished

def Basic_unittest_C(C):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]    
    # C OK
    if type(C) != type(randomMatrix):
        raise TypeError("C should be a numpy array")
    if len(C.shape) != 2:
        raise ValueError("C should be an n x n symmetric matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C should be an n x n symmetric matrix")
    if not np.array_equal(C, C.T):
        raise ValueError("C should be an n x n symmetric matrix")
    for c in C:
        for j in c:
            if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
                raise TypeError(f"Value in C sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
# Finished

def Basic_unittest_X(X):
    randomMatrix = np.array([[1]])
    randomNumpyInt = np.array([1])[0]
    randomNumpyFloat = np.array([11.1])[0]
    # X OK
    if type(X) != type(randomMatrix):
        raise TypeError("X should be a numpy array")
    if len(X.shape) != 2:
        raise ValueError("X should be an n x 1 vector")
    if X.shape[1] != 1:
        raise ValueError("X should be an n x 1 vector")
    for x in X:
        j = x[0]
        if type(j) != type(randomNumpyInt) and type(j) != type(randomNumpyFloat):
            raise TypeError(f"Value in X sould be either {type(randomNumpyInt)} or {type(randomNumpyFloat)}")
# Finished

'''We can calculate the likelihood of obtaining
the data under our Brownian motion model using a standard formula for the
likelihood of drawing from a multivariate normal distribution (Harmon & Open Textbook Library, 2019)'''
# Reference is:
# Harmon, L. J. & Open Textbook Library. (2019). Phylogenetic Comparative Methods. https://open.umn.edu/opentextbooks/textbooks/691 
# Section 4.3: Estimating rates using maximum likelihood (eq. 4.5)
# Since it is an implementation of a mathematic formular, please compare our implementation with that mathematic formular in the reference.
def Brownian_motion_likelihood(X,Z0, deltaSquare, C):
    Basic_unittest(X,Z0, deltaSquare, C)
    X = X * 1.0
    Z0 = Z0 * 1.0
    deltaSquare = deltaSquare * 1.0
    C = C * 1.0
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    Z0_vector = Z0 * one
    XSubZ0_vector = X - Z0_vector
    # This is because pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
    temp = np.dot(np.dot(np.transpose(XSubZ0_vector), pinv(deltaSquare * C)), XSubZ0_vector)
    numerator = math.exp(-1/2 * temp) # This is correct
    try:
        denominator = math.sqrt(((2 * math.pi) ** len(X)) * det(deltaSquare * C)) # This is correct
        #denominator = math.sqrt(((2 * math.pi) ** len(X)) * deltaSquare**len(C) * det(C))
    except FloatingPointError: # OK to have this error and continue executing next time
        return 0
    likelihood = numerator / denominator
    return likelihood # correct

# ln of Brownian motion likelihood
def Ln_Brownian_motion_likelihood(X,Z0, deltaSquare, C):
    Basic_unittest(X,Z0, deltaSquare, C)
    X = X * 1.0
    Z0 = Z0 * 1.0
    deltaSquare = deltaSquare * 1.0
    C = C * 1.0
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    Z0_vector = Z0 * one
    XSubZ0_vector = X - Z0_vector
    # This is because pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
    temp = np.dot(np.dot(np.transpose(XSubZ0_vector), pinv(deltaSquare * C)), XSubZ0_vector)
    front = -1/2 * temp
    end = 1/2 * (len(X)*np.log(2 * math.pi) + slogdet(deltaSquare * C)[1])
    return (front - end)[0][0]


'''In this case, the maximum-likelihood estimate for each of these two parameters can be calculated analytically (Harmon & Open Textbook Library, 2019)'''
# Reference is:
# Harmon, L. J. & Open Textbook Library. (2019). Phylogenetic Comparative Methods. https://open.umn.edu/opentextbooks/textbooks/691 
# Section 4.3: Estimating rates using maximum likelihood (eq. 4.7) and (eq. 4.8)
# Since it is an implementation of mathematic formulars, please compare our implementation with mathematic formulars in the reference.
def Brownian_motion_maximumlikelihood(X, C):
    Basic_unittest_X_C(X, C)
    X = X * 1.0
    C = C * 1.0
    # one is n x 1 vector of 1
    one = np.full((len(X), 1), 1)
    z0hat_front = pinv(one.T @ pinv(C) @ one)
    z0hat_end = one.T @ pinv(C) @ X
    # estimated root state for the character
    z0hat = z0hat_front * z0hat_end

    # maximum likelihood delta square
    numerator = (X - z0hat * one).T @ pinv(C) @ (X - z0hat * one)
    denominator = len(X)
    # estimated net rate of evolution
    deltaSquarehat = numerator / denominator
    # print(deltaSquarehat)
    return z0hat, deltaSquarehat

# Based on Pagel's lambda to transform the phylogenetic variance-covariance matrix.
# compresses internal branches while leaving the tip branches of the tree unaffected
# Reference is:
# Harmon, L. J. & Open Textbook Library. (2019). Phylogenetic Comparative Methods. https://open.umn.edu/opentextbooks/textbooks/691 
# Section 6.2: Transforming the evolutionary variancecovariance matrix (Equation 6.2)
# Since it is an implementation of a mathematic formular, please compare our implementation with that mathematic formular in the reference.
def lambdaCovarience(C, lambdaVal):
    Basic_unittest_C(C)
    if type(lambdaVal) != type(1) and type(lambdaVal) != type(0.5):
        raise TypeError("lambdaVal should be either an integer or a float")
    # 0 <= lambda <= 1 
    if (lambdaVal < 0) or (lambdaVal > 1):
        raise ValueError("Lambda value: 0 <= lambda <= 1")
    C = C * 1.0
    n = len(C)
    for i in range(0,n):
        for j in range(0,n):
            # Off diagonal times lambda
            if i != j:
                C[i][j] = C[i][j] * lambdaVal
    return C

# Compute MLE for a given lambda value
def Pagel_lambda_MLE(X, C, lambdaVal):
    Basic_unittest_X_C(X, C)
    if type(lambdaVal) != type(1) and type(lambdaVal) != type(0.5):
        raise TypeError("lambdaVal should be either an integer or a float")
    # 0 <= lambda <= 1 
    if (lambdaVal < 0) or (lambdaVal > 1):
        raise ValueError("Lambda value: 0 <= lambda <= 1")
    X = X * 1.0
    C = C * 1.0
    # Compute new covarience matrix
    C_lambda = lambdaCovarience(C, lambdaVal)
    z0hat, deltaSquarehat = Brownian_motion_maximumlikelihood(X, C_lambda)
    # Compute ln likelihood
    Pagel_likelihood = Ln_Brownian_motion_likelihood(X,z0hat,deltaSquarehat, C_lambda)

    return Pagel_likelihood, z0hat, deltaSquarehat, lambdaVal

# Searching with different step sizes. Finding out lambda's value to maximize likelihood
def Found_Pagel_Maximumlikelihood(X, tree, stepSize, startSearch = 0, EndSearch = 1):
    Basic_unittest_X(X)
    # Just used to get type
    defaultTree = Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);")
    if type(tree) != type(defaultTree):
        raise TypeError(f"tree type should be {type(defaultTree)}")
    treeLen = len(tree)
    if X.shape[0] != treeLen:
        raise ValueError("Wrong X and tree input, dimension does not match")
    # stepSize check
    if type(stepSize) != type(1) and type(stepSize) != type(0.5):
        raise TypeError("stepSize should be either an integer or a float")
    if (stepSize <= 0):
        raise ValueError("stepSize value: 0 < stepSize")
    # startSearch check
    if type(startSearch) != type(1) and type(startSearch) != type(0.5):
        raise TypeError("startSearch should be either an integer or a float")
    if (startSearch < 0) or (startSearch > 1):
        raise ValueError("startSearch value: 0 <= startSearch <= 1")
    # EndSearch check
    if type(EndSearch) != type(1) and type(EndSearch) != type(0.5):
        raise TypeError("EndSearch should be either an integer or a float")
    if (EndSearch < 0) or (EndSearch > 1):
        raise ValueError("EndSearch value: 0 <= EndSearch <= 1")
    if startSearch > EndSearch:
        raise ValueError("startSearch shoud be smaller than or equal to EndSearch")
    # OK
    X = X * 1.0
    '''One can in principle use some values of lambda greater than one on most variance-covariance
        matrices, although many values of lambda > 1 result in matrices that are not valid
        variance-covariance matrices and/or do not correspond with any phylogenetic
        tree transformation. For this reason I recommend that lambda be limited to values
        between 0 and 1.'''
    # Initialization
    lambdaVal = startSearch
    maxlikelihood = -math.inf
    maxlikelihood_lambda = -math.inf
    # Record all likelihood and lambda value
    likelihoodSave = []
    lambdaValSave = []
    # Try different lambda values and try to find its corresponding MLE
    while lambdaVal <= EndSearch:
        print(lambdaVal)
        # Recompute C every time because it will be overwrited
        C = Covariance(tree)
        tmp_likelihood,tmp_z0hat, tmp_deltaSquarehat, tmp_lambdaVal= Pagel_lambda_MLE(X, C, lambdaVal)
        # If tmp value is larger
        if maxlikelihood < tmp_likelihood:
            maxlikelihood = tmp_likelihood
            maxlikelihood_lambda = lambdaVal
        likelihoodSave.append(tmp_likelihood)
        lambdaValSave.append(lambdaVal)
        lambdaVal += stepSize
    # Return all of them
    return maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave

# This function will return covariance matrix
def Covariance(bac_tree):
    defaultTree = Tree("(A:1,(B:1,(C:1,D:1):0.5):0.5);")
    if type(bac_tree) != type(defaultTree):
        raise TypeError(f"bac_tree type should be {type(defaultTree)}")
    # OK
    # Create n by n matrix Make sure pruning the tree or it maybe too large and have error occur.
    C = np.zeros(shape=(len(bac_tree), len(bac_tree)))
    # Used to tranverse through the matrix
    i_counter = -1
    j_counter = -1
    # Tranverse through all leaves
    for leaf_i in bac_tree:
        # Corresponding index
        i_counter += 1
        # Tranverse through all leaves
        for leaf_j in bac_tree:
            j_counter += 1
            # If they are the same leaf
            if leaf_i == leaf_j:
                # Covariance is just its distance to the root
                C[i_counter][j_counter] = leaf_i.get_distance(bac_tree)
            else:
                # Get their first common ancestor and compute its distance to root
                commonAncestor = leaf_i.get_common_ancestor(leaf_j)
                C[i_counter][j_counter] = commonAncestor.get_distance(bac_tree)
        j_counter = -1
    return C

def traitsColumnReturn(df, traits_name):
    traits = list(df.loc[:,f'{traits_name}'])
    X = []
    for i in traits:
        X.append([i])
    X = np.array(X)
    return X


# Step size 0.01
def Bacteria_Pagel():
    # Load trees
    # bac_tree is the root of the tree
    bac_tree = Tree('./Data/IGG_v1.0_bact_22515.tre')
    # Find correspondence
    IGG_Haojun = pd.read_csv("./Data/igg_haojun.csv")
    IGG_Haojun_altid2genomeid = IGG_Haojun.loc[:, ['genome_id', 'species_alt_id']]

    out_all = pd.read_csv("./Data/out-all-2.csv")
    out_all_genome_id = out_all.iloc[:,0]
    
    headers = list(IGG_Haojun_altid2genomeid.columns)
    translate = pd.DataFrame(columns=headers)
    # find correspondance
    for genome_id in out_all_genome_id:
        row = IGG_Haojun_altid2genomeid.loc[IGG_Haojun_altid2genomeid['genome_id'] == genome_id]
        # translate = translate.append(row,ignore_index=True)
        translate = pd.concat([translate, row],ignore_index=True)

    # Slice all samples
    sample_id = list(out_all.columns)
    sample_id = sample_id[1:]
    # save for all maximum likelihood lambda in a format of dictionary of dictionary
    maximumlikelihoodSave = {}
    maximumlikelihoodOtherSave = {}
    for i in sample_id:
        out_all_new_sample = out_all.loc[:,i]
        tmp_translate = translate
        out = tmp_translate.join(out_all_new_sample)
        keep = out[pd.notnull(out[i])]
        keep = keep.reset_index(drop=True)
        keep_list = list(keep.iloc[:,1])
        # Make the become string
        for j in range(0, len(keep_list)):
            keep_list[j] = str(keep_list[j])
        # Create a tree to do pruning operation
        bac_tree_op = bac_tree.copy()
        bac_tree_op.prune(keep_list, preserve_branch_length=True)
        # Compute covarience
        C = Covariance(bac_tree_op)
        # Reorder the feature
        reorder_header = list(keep.columns)
        reorder = pd.DataFrame(columns=reorder_header)
        for leaf in bac_tree_op:
            row = keep.loc[keep['species_alt_id'] == int(leaf.name)]
            # reorder = reorder.append(row,ignore_index=True)
            reorder = pd.concat([reorder, row],ignore_index=True)

        # Log2(PTR)
        X = traitsColumnReturn(reorder, i)
        # PTR
        X = np.exp2(X)
        # Can be negative but be careful with the domain, I do not advise to do so because it is meaningless and may cause math domain error (math.sqrt(negative value))
        # Greater than 1 also has math domain error.
        # ln maximumlikelihood
        # Modified
        #maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, bac_tree_op, 0.01,startSearch=0,EndSearch=1)
        maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, bac_tree_op, 0.01,startSearch=-1,EndSearch=1)

        if maxlikelihood_lambda != None:
            print(f'Sample: {i}; Ln Maximum likelihood: {maxlikelihood}; Number of leaves: {len(keep)}; Lambda: {round(maxlikelihood_lambda,2)}')
        else:
            print(f'Sample: {i}; Ln Maximum likelihood: {maxlikelihood}; Number of leaves: {len(keep)}; Lambda: {maxlikelihood_lambda}')

        if len(keep) not in maximumlikelihoodSave.keys():
            maximumlikelihoodSave[len(keep)] = {i:maxlikelihood_lambda}
            maximumlikelihoodOtherSave[len(keep)] = {i:[maxlikelihood, likelihoodSave, lambdaValSave]}
        else:
            maximumlikelihoodSave[len(keep)][i] = maxlikelihood_lambda
            maximumlikelihoodOtherSave[len(keep)][i] = [maxlikelihood, likelihoodSave, lambdaValSave]
        
        # if maxlikelihood_lambda != 0 and maxlikelihood_lambda != None:
            # try:
        # fig, ax = plt.subplots(1, 1, figsize=(10,10))
        plt.plot(lambdaValSave, likelihoodSave, 'ro')
        plt.xlabel("Lambda Value")
        plt.ylabel("Likelihood")
        plt.title(f'Pagel\'s Lambda: Lambda-Likelihood Plot \n \n\
            Sample: {i}; Maximum likelihood: {maxlikelihood}; \n \n Number of leaves: {len(keep)}; Lambda: {round(maxlikelihood_lambda,2)}', fontweight='bold', fontsize=12)
        # plt.xlim([0, 1]) Modified
        plt.xlim([-1, 1])
        # ax.savefig(f"./Plots/{i}%{len(keep)}.png", bbox_inches = 'tight')
        # ax.savefig(f"./Plots_PTR/{i}%{len(keep)}.png", bbox_inches = 'tight')
        # plt.savefig(f"./Updated_Plots_PTR_LnMax/{i}%{len(keep)}.png", bbox_inches = 'tight') Modified
        plt.savefig(f"./Updated_Plots_PTR_LnMax_-1to1/{i}%{len(keep)}.png", bbox_inches = 'tight')
        
        plt.clf()
            # except Exception as e:
            #     print("+++++++++++++++++++++++++++++++")
            #     # print(lambdaValSave)
            #     # print(likelihoodSave)
            #     print(e)
            #     print("+++++++++++++++++++++++++++++++")

    return maximumlikelihoodSave, maximumlikelihoodOtherSave


# maximumlikelihoodSave needs a dictionary of dictionary
# numBins needs how many bins you want
def Lambda_Hist(maximumlikelihoodSave, numBins):
    lambdaList = []
    for leaf_num in maximumlikelihoodSave.keys():
        for sampleVal in maximumlikelihoodSave[leaf_num].keys():
            if maximumlikelihoodSave[leaf_num][sampleVal] != None:
                lambdaList.append(maximumlikelihoodSave[leaf_num][sampleVal])
    plt.hist(lambdaList, numBins)
    plt.title(f'Histogram of Maximum Likelihood Lambda \n \n Number of Valid Samples: {len(lambdaList)}', \
             fontweight='bold', fontsize=12)
    plt.xlabel("Lambda Values")
    plt.ylabel("Number of Samples")
    # plt.savefig(f"./Plots/Lambda_Samples_Histogram.png", bbox_inches = 'tight')
    # plt.savefig(f"./Plots_PTR/Lambda_Samples_Histogram.png", bbox_inches = 'tight')
    #plt.savefig(f"./Updated_Plots_PTR_LnMax/Lambda_Samples_Histogram.png", bbox_inches = 'tight') Modified
    plt.savefig(f"./Updated_Plots_PTR_LnMax_-1to1/Lambda_Samples_Histogram.png", bbox_inches = 'tight')
    plt.clf()
    #plt.show()

def leaves_lambda(maximumlikelihoodSave):
    leaves = []
    lambdas = []
    # {leaves: {sample:lambda}}
    for i in maximumlikelihoodSave.keys():
        for j in maximumlikelihoodSave[i].keys():
            if maximumlikelihoodSave[i][j] != None:
                leaves.append(i)
                lambdas.append(maximumlikelihoodSave[i][j])
    plt.plot(leaves, lambdas, 'ro', alpha=0.5)
    plt.xlabel("Number of Leaves")
    plt.ylabel("Lambda Values")
    plt.title(f"Pagel\'s Lambda -- Leaves-Lambda Distribution Plots \n \n Number of Samples: {len(lambdas)}")
    # plt.savefig(f"./Plots/Leaves_lambdas.png", bbox_inches = 'tight')
    # plt.savefig(f"./Plots_PTR/Leaves_lambdas.png", bbox_inches = 'tight')
    # plt.savefig(f"./Updated_Plots_PTR_LnMax/Leaves_lambdas.png", bbox_inches = 'tight') Modified
    plt.savefig(f"./Updated_Plots_PTR_LnMax_-1to1/Leaves_lambdas.png", bbox_inches = 'tight')
    
    plt.clf()
    #plt.show()

def lambda_subplot(maximumlikelihoodSave, maximumlikelihoodOtherSave, width = 0.1, height = 0.1 , opacity = 0.5):
    leaves = []
    lambdas = []
    # {leaves: {sample:lambda}}
    for i in maximumlikelihoodSave.keys():
        for j in maximumlikelihoodSave[i].keys():
            if maximumlikelihoodSave[i][j] != None:
                leaves.append(i)
                lambdas.append(maximumlikelihoodSave[i][j])
    fig, ax = plt.subplots()
    ax.plot(leaves, lambdas, 'ro')
    ax.set_xlabel("Number of Leaves")
    ax.set_ylabel("Lambda Values")
    ax.set_title(f"Pagel\'s Lambda -- Leaves-Lambda Distribution Plots \n \n Number of Samples: {len(lambdas)}")
    # maximumlikelihoodOtherSave[len(keep)] = {i:[maxlikelihood, likelihoodSave, lambdaValSave]}
    for i in maximumlikelihoodOtherSave.keys():
        for j in maximumlikelihoodOtherSave[i].keys():
            if maximumlikelihoodSave[i][j] != None:
                x_trans = (i - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
                y_trans = (maximumlikelihoodSave[i][j] - ax.get_ylim()[0])/ (ax.get_ylim()[1] - ax.get_ylim()[0])
                axins = ax.inset_axes([x_trans, y_trans, width, height])
                axins.plot(maximumlikelihoodOtherSave[i][j][2], maximumlikelihoodOtherSave[i][j][1])
                axins.patch.set_alpha(opacity)
    # fig.savefig(f"./Updated_Plots_PTR_LnMax/LeavesLambdaDistribution.png", bbox_inches = 'tight') Modified
    fig.savefig(f"./Updated_Plots_PTR_LnMax_-1to1/LeavesLambdaDistribution.png", bbox_inches = 'tight')
    
    plt.show()

# Plot
# maximumlikelihoodSave, maximumlikelihoodOtherSave = Bacteria_Pagel()
# Lambda_Hist(maximumlikelihoodSave, 10)
# leaves_lambda(maximumlikelihoodSave)
# lambda_subplot(maximumlikelihoodSave, maximumlikelihoodOtherSave)

'''The column names have the format "{patient_id}_K{visit_num}0", i.e. "EP003595_K10" means "patient EP003595" visit 1. 
The row names are Greengenes IDs, and they should map to the Greengenes 12_10 tree I showed you. 
I'm not sure what the actual time intervals are. You can use subsequent visits for the same patient to look at general growth rates.'''
def MOMS_PI_Dataset_Pagel():
    # Process data first
    with open("C:/Users/12533/Desktop/Fall 2022/COMSE 6901/Fall 2022/MOMS-PI dataset/mixture/16s/16s_tables.pkl", "rb") as f:
        tables = pickle.load(f)
    # Top 10% densest clades
    top10_clades = (tables['MCKD'].to_dataframe() > 0).sum(axis=1).sort_values()[-375:]
    # Normalization: make every sample sum to 1
    mckd_normed = tables['MCKD'].norm(axis='sample', inplace=False)
    mckd_normed_top10 = mckd_normed.filter(top10_clades.index, axis='observation', inplace=False) # Filter to top 10% densest clades
    
    colnames_split = [x.split("_") for x in mckd_normed_top10.to_dataframe().columns]
    colnames_split2 = [(patient, int(visit[1:-1])) for (patient, visit) in colnames_split]
    mckd_normed_top10_reindexed = mckd_normed_top10.copy().to_dataframe()
    mckd_normed_top10_reindexed.columns = pd.MultiIndex.from_tuples(colnames_split2) # Make a multiindex with patient and visit
    # Sort by patient and visit
    mckd_normed_top10_reindexed = mckd_normed_top10_reindexed.sort_index(axis=1, level=1) # Sort by visit
    mckd_normed_top10_reindexed = mckd_normed_top10_reindexed.sort_index(axis=1, level=0) # Sort by patient
    
    mckd_diffs = mckd_normed_top10_reindexed.groupby(level=0, axis=1).diff() # Group by patient, then take difference between visits
    
    # Drop the columns where all elements are NaN:
    mckd_diffs = mckd_diffs.dropna(axis=1, how='all')
    # OLD CODE Modifiy
    
    # Load trees
    # bac_tree is the root of the tree
    bac_tree = Tree('C:/Users/12533/Desktop/Fall 2022/COMSE 6901/Fall 2022/MOMS-PI dataset/gg_13_5_otus_99_annotated_Newick.tree')

    # save for all maximum likelihood lambda in a format of dictionary of dictionary
    maximumlikelihoodSave = {}
    maximumlikelihoodOtherSave = {}
    
    for patientID, visitTimes in mckd_diffs.columns.tolist():
        keep = pd.DataFrame(mckd_diffs[patientID][visitTimes])
        keep_list = []
        for i in keep.index:
            keep_list.append(str(i))
        # print(keep_list)
        # return
        # Create a tree to do pruning operation
        bac_tree_op = bac_tree.copy()
        bac_tree_op.prune(keep_list, preserve_branch_length=True)

        # Reorder the feature
        reorder = pd.DataFrame()
        for leaf in bac_tree_op:
            row = keep.loc[str(leaf.name)]
            # reorder = reorder.append(row,ignore_index=True)
            reorder = pd.concat([reorder, row],ignore_index=True)

        # print(reorder)
        X = reorder.iloc[:, 0].to_numpy()
        X = np.array([X])
        X = X.T
        # Can be negative but be careful with the domain, I do not advise to do so because it is meaningless and may cause math domain error (math.sqrt(negative value))
        # Greater than 1 also has math domain error.
        # ln maximumlikelihood
        # Modified
        maxlikelihood, maxlikelihood_lambda, likelihoodSave, lambdaValSave=Found_Pagel_Maximumlikelihood(X, bac_tree_op, 0.01,startSearch=0,EndSearch=1)
        # {patient_id}_K{visit_num}0
        if maxlikelihood_lambda != None:
            print(f'Sample: {patientID}_K{visitTimes}0; Ln Maximum likelihood: {maxlikelihood}; Number of leaves: {len(keep.index)}; Lambda: {round(maxlikelihood_lambda, 2)}')
        else:
            print(f'Sample: {patientID}_K{visitTimes}0; Ln Maximum likelihood: {maxlikelihood}; Number of leaves: {len(keep.index)}; Lambda: {maxlikelihood_lambda}')

        if len(keep.index) not in maximumlikelihoodSave.keys():
            maximumlikelihoodSave[len(keep.index)] = {f'{patientID}_K{visitTimes}0':maxlikelihood_lambda}
            maximumlikelihoodOtherSave[len(keep.index)] = {f'{patientID}_K{visitTimes}0':[maxlikelihood, likelihoodSave, lambdaValSave]}
        else:
            maximumlikelihoodSave[len(keep.index)][f'{patientID}_K{visitTimes}0'] = maxlikelihood_lambda
            maximumlikelihoodOtherSave[len(keep.index)][f'{patientID}_K{visitTimes}0'] = [maxlikelihood, likelihoodSave, lambdaValSave]
        

        plt.plot(lambdaValSave, likelihoodSave, 'ro')
        plt.xlabel("Lambda Value")
        plt.ylabel("Likelihood")
        plt.title(f'Pagel\'s Lambda: Lambda-Likelihood Plot \n \n\
            Sample: {patientID}_K{visitTimes}0; Ln Maximum likelihood: {maxlikelihood}; \n \n Number of leaves: {len(keep.index)}; \
                Lambda: {round(maxlikelihood_lambda,2)}', fontweight='bold', fontsize=12)
        plt.xlim([0, 1])
        plt.savefig(f"../Plots_MOMS-PI_LnMax/{patientID}_K{visitTimes}0%{len(keep.index)}.png", bbox_inches = 'tight')
        
        plt.clf()
    return maximumlikelihoodSave, maximumlikelihoodOtherSave
MOMS_PI_Dataset_Pagel()