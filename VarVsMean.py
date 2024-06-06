import numpy as np
import random as random
import time as tim
import math
from sklearn.linear_model import LinearRegression
import copy


class scatterData:
    def __init__(self, time, var, mn):
        self.time = time
        self.var = var
        self.mn = mn

class distData:
    def __init__(self, counts, whichBins, bins):
        self.counts = counts
        self.whichBins = whichBins
        self.bins = bins
        self.countsSelect = []

def VarVsMean(data, times, params:dict):
    '''
    Here we add general comments for the function. Like inputs, outputs and so on
    '''

    # Handle params input
    boxWidth = params.get('boxWidth', 80) # width of the sliding window in which the counts are made
    matchReps = params.get('matchReps', 10) # number of random choices regarding which points to throw away when matching distributions
    binSpacing = params.get('binSpacing', 0.25) # bin width when computing distributions of mean counts
    alignTime = params.get('alignTime', 0) # time of event that data are aligned to (in the output structure, times will be expressed relative to this)
    initRand = params.get('initRand', 1)
    weightedRegression = params.get('weightedRegression', True)
    includeVarMinusMean = params.get('includeVarMinusMean', True)
    lenTimes = len(times)
    lenData = len(data)
    # Init random seed
    if initRand:
        random.seed(int(tim.time()))
        np.random.seed(int(tim.time()))

    # Check inputs is correct (maybe)

    # Init variables
    weightingEpsilon = boxWidth/1000
    ScatterDataAll = []
    maxRate = 0
    trialCount = np.zeros(lenData)

    # DO CONVOLUTION WITH SLIDING WINDOW.
    # COMPUTE THE MEAN AND VARIANCE OF THE COUNT FOR EACH TIME & EACH NEURON/CONDITION 
    # Init ScatterDataAll
    for t in times:
        time = t-alignTime
        var = np.zeros(lenData)
        mn = np.zeros(lenData)
        ScatterDataAll.append(scatterData(time, var, mn))

    # Do calculations
    TStart = times - math.floor(boxWidth/2) + 1
    TEnd = times + math.ceil(boxWidth/2) + 1 

    print(TStart)
    print(TEnd)

    maxRate = 0
    for cond in range(0,lenData):
        data[cond] = np.asmatrix(data[cond])
        csum = np.cumsum((data[cond][:,TStart[0]-1:TEnd[-1]+1]), axis=1)
        count = csum[:,TEnd-TStart[0]] - csum[:,TStart-TStart[0]]

        varTemp = np.apply_along_axis(np.var,0,count)
        mnTemp = np.apply_along_axis(np.mean,0,count)

        for t in range(0, lenTimes):
            ScatterDataAll[t].var[cond] = varTemp[t]
            ScatterDataAll[t].mn[cond] = mnTemp[t]
            
        maxRate = max(maxRate, np.max(mnTemp))
        trialCount[cond] = np.shape(data[cond])[0]
        

    if maxRate < 0:
        print("error was encountered")
        return {}
    # COMPUTE FIRING RATE DISTRIBUTIONS AT EACH TIME TO BE TESTED
    # This is necessary even if we aren't downselecting, to provide the user with bin and dist info
        
    meanRates = np.zeros((lenTimes,2))  # initialize to save time
    bins = np.arange(0,maxRate+binSpacing, binSpacing)
    stdOfDeltaRate = np.zeros(lenTimes)
    distDataAll = []
    lenBins = len(bins)
    binMatrix = np.zeros((lenBins,lenTimes))
    for t in range(0,lenTimes):
        meanRates[t,0] = 1000 / boxWidth * np.mean(ScatterDataAll[t].mn) # convert to spikes/s (ONLY for the rate)
        stdOfDeltaRate[t] = 1000 / boxWidth * np.std(np.subtract(ScatterDataAll[t].mn, ScatterDataAll[0].mn)); # standard deviation of means
        whichBins = np.digitize(ScatterDataAll[t].mn, bins) # Use np.digitize here 
        counts = getCount(whichBins, bins)
        distDataAll.append(distData(counts,whichBins,bins))
        #print(counts)
        binMatrix[:,t] = counts
                  
        
    targetCount = np.min(binMatrix, axis = 1);  #takes min count, across the measurement times, for each bin.
    # COMPUTE SLOPE VERSUS TIME FOR ENTIRE DATA SET (no distribution matching yet)
    
    slopesAll = []
    slopesAll_95CIs = []
    VMMall = []
    VMMall_95CIs = []
    sumSlopes = []
    sumCIs = []
    regWeights = np.array([])
    for t in range(0,lenTimes):
        if weightedRegression:
            regWeights = trialCount / (ScatterDataAll[t].mn + weightingEpsilon) ** 2
        else:
            regWeights = np.ones(np.shape(ScatterDataAll[t].mn))
        b, stdB = lscov(ScatterDataAll[t].mn, ScatterDataAll[t].var, regWeights)
        Bint = np.array([b-2.0*stdB, b+2.0*stdB])

        slopesAll.append(b)
        slopesAll_95CIs.append(Bint)

        if includeVarMinusMean:
            meanTemp = np.mean(ScatterDataAll[t].var - ScatterDataAll[t].mn)
            semTemp = np.std(ScatterDataAll[t].var - ScatterDataAll[t].mn) / math.sqrt(len(ScatterDataAll[t].var))
            VMMall.append(meanTemp)
            tempArr = np.array([-2.0*semTemp, 2*semTemp])
            VMMall_95CIs.append(meanTemp + tempArr)
    
    
    for t in range(0,lenTimes):
        distDataAll[t].countsSelect = np.zeros((lenBins))

    # NOW DO THE DISTRIBUTION MATCHING
    if matchReps > 0:
        ScatterDataAllSelect = copy.deepcopy(ScatterDataAll)

        sumSlopes = np.zeros((lenTimes,1))
        sumCIs = np.zeros((lenTimes,2))

        if includeVarMinusMean:
            sumDiffs = np.zeros((lenTimes,1))
            sumDiffCIs = np.zeros((lenTimes,2))

        for rep in range(0,matchReps):
            for t in range(0,lenTimes):
                toKeep = np.array([])
                tempBin = distDataAll[t].whichBins
                for b in range(0,lenBins-1):
                    thisBin = np.array([i for i, val in enumerate(tempBin) if val == (b+1)])
                    thisBin = np.random.permutation(thisBin)
                    indexCount = int(targetCount[b])
                    if indexCount != 0:
                        toKeep = np.concatenate((toKeep,thisBin[0:indexCount]))
                    distDataAll[t].countsSelect[b] = len(thisBin[0:indexCount])
                distDataAll[t].countsSelect[lenBins-1] = 0
                if len(toKeep) < 5:
                    print("Problem in VarVsMean")
                    print("No data survuved mean matching")
                    return {}
                toKeep = toKeep.astype(int)
                mnsThisRep = ScatterDataAll[t].mn[toKeep]
                varsThisRep = ScatterDataAll[t].var[toKeep]

                if weightedRegression:
                    regWeights = trialCount[toKeep] / (mnsThisRep + weightingEpsilon)**2
                else:
                    regWeights = np.ones(np.shape(mnsThisRep))
                b, stdB = lscov(mnsThisRep,varsThisRep, regWeights)
                Bint = np.reshape(np.array([b-2*stdB, b+2*stdB]),(2,))
                
                sumSlopes[t] = sumSlopes[t] + b
                sumCIs[t] = sumCIs[t] + Bint

                if rep == 0:
                    ScatterDataAllSelect[t].mn = mnsThisRep
                    ScatterDataAllSelect[t].var = varsThisRep

                meanRates[t, 1] = meanRates[t, 1] + 1000/boxWidth*np.mean(mnsThisRep)/matchReps

                if includeVarMinusMean:
                    meanTemp = np.mean(varsThisRep - mnsThisRep)
                    semTemp = np.std(varsThisRep - mnsThisRep) / np.sqrt(len(varsThisRep))
                    sumDiffs[t] = sumDiffs[t] + meanTemp
                    tempArr = np.array([-2.0*semTemp, 2*semTemp])
                    sumDiffCIs[t,:] = sumDiffCIs[t,:] + meanTemp + tempArr

        slopes = sumSlopes / matchReps
        slopes_95CIs = sumCIs / matchReps
        if includeVarMinusMean:
            diffs = sumDiffs / matchReps
            diffs_95CIs = sumDiffCIs / matchReps
        
    result = {"FanoFactorAll":slopesAll, "FanoAll_95CIs":slopesAll_95CIs, "scatterDataAll":ScatterDataAll,"meanRateAll":meanRates[:,0], "stdOfDeltaRate":stdOfDeltaRate,"distData":distDataAll,"times":(times-alignTime)}

    if matchReps > 0:
        match_rep_data = {"FanoFactor":slopes,"Fano_95CIs":slopes_95CIs, "scatterData":ScatterDataAllSelect,"meanRateSelect":meanRates[:,1]}
        result.update(match_rep_data)
        if includeVarMinusMean:
            varMinusMean_data = {"varMinusMean":diffs, "VMM_95CIs":diffs_95CIs,"VMMall":VMMall,"VMMall_95Cis":VMMall_95CIs}
            result.update(varMinusMean_data)
    
    return result

def lscov(mn,var,weights):
    
    X = np.array(mn)
    y = np.array(var)
    w = np.array(weights)
    diag_w = np.diag(w)
    if X.ndim < 2:
        numDim = 1
        X_weighted = np.dot(np.sqrt(diag_w), X)
        y_weighted = np.dot(np.sqrt(diag_w), y)
        b, residuals, _, _ = np.linalg.lstsq(X_weighted.reshape(-1, 1), y_weighted, rcond=None)
        
        ## Get A and B vectors
        A = np.sqrt(w) * X
        B = np.sqrt(w) * y

        tempMat = np.zeros((A.shape[0],A.shape[0]))
        tempMat[:,0] = A
        Q, R = np.linalg.qr(tempMat)
        Q = Q[:,0]
        R = R[0,0]
        z = (Q @ B)
        

        ## Calculate MSE
        res = B - Q*z
        conRes = np.conj(res)
        MSE = np.sum(res*conRes) / (A.shape[0]-1)

        ## Calculate R and std
        Rinv = R**-1
        stdX = np.sqrt(Rinv*Rinv*MSE)
        return b[0], stdX


    else:
        diag_w = np.zeros((len(weights),len(weights)), float)
        np.fill_diagonal(diag_w,w)
        Aw = X * np.sqrt(w[:,np.newaxis])
        Bw = y * np.sqrt(w)
        b, residuals, _, _  = np.linalg.lstsq(Aw, Bw, rcond=None)
        
        # Compute the covariance matrix
        residual_variance = np.sum(residuals)

        numDim = X.shape[1]
        
        divd =  (len(y) - numDim)
        residual_variance = residual_variance / divd
        cov_matrix = np.linalg.inv(Aw.T @ Aw) * residual_variance
        
        # Extract standard errors from diagonal of covariance matrix
        std_errors = np.sqrt(np.diag(cov_matrix))

        return b, std_errors

def getMinCounts(dist):
    minArray = []
    for line in dist:
        minArray.append(min(line))
    return minArray

def getCount(whichBin, bins):
    counts = np.zeros(np.shape(bins))
    for val in whichBin:
        counts[val-1] += 1
    return counts
