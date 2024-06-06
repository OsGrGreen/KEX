from json import JSONEncoder
import json

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from newPlotFano import newPlotFano
from scipy.stats import linregress
from run_project import NumpyArrayEncoder
import seaborn as sns


typeToType = {0:"grey", 1:"VISp", 2:"VISal",3:"VISpm", 4:"VISrl",5:"VISl"}
typeToNum = {'grey': 0, 'VISp': 1, 'VISal': 2, 'VISpm': 3, 'VISrl': 4, 'VISl': 5}
typeToName = {'grey': 'G', 'VISp': 'Vp', 'VISal': 'Va', 'VISpm': 'Vm', 'VISrl': 'Vr', 'VISl': 'Vl'}
# ['grey' 'VISp' 'VISal' 'VISpm' 'VISrl' 'VISl']
differentFromTheory = {}
noDiffThreshold = 0.015

def start_parsing(shouldPlot, heatMaps):
    try:
        with open("processedData/pross_data1.json", "r") as read_file:
            
            input_data = json.load(read_file)
    except FileNotFoundError:
        print("File not found!")
        return
    

    print("Has loaded file")
    keysList = list(input_data.keys())
    curType = 0

    times = input_data[keysList[0]][list(input_data[keysList[0]].keys())[0]]['res']['times']
    print("Times are: ", times)
    numTimes = len(times)

    neuronNameTranslationTable = []

    ## SNS setup
    col = sns.color_palette("coolwarm", as_cmap=True)
    FFcolor = [0]*3
    figHeat, axesHeat = plt.subplots(2, 3, sharex=True)
    fanoAllAll = np.zeros((len(keysList),len(input_data[keysList[0]].keys()),numTimes))
    fanoAllMatched = np.zeros((len(keysList),len(input_data[keysList[0]].keys()),numTimes))
    cbar_ax = figHeat.add_axes([.95, .3, .03, .4])
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([])
    typeUnits = []
    keyToUnits = {}
    i = 0
    j = 0
    for type in keysList:
        typeValues = input_data[type]
        units = list(typeValues.keys())
        keyToUnits[type] = units
        counter = 0

        for unit in units:
            res = typeValues[unit]['res']
            fanoAllAll[typeToNum[type]][counter] = res['FanoFactorAll']
            fanoAllMatched[typeToNum[type]][counter] = np.column_stack(res['FanoFactor'])
            neuronNameTranslationTable.append((type+unit,typeToName[type]+str(counter)))
            if shouldPlot:
                newPlotFano(res, False, typeToName[type]+str(counter))
            counter += 1 

        plotVals = fanoAllMatched[typeToNum[type]] #- fanoAllMatched[typeToNum[type]][:, 0].reshape(-1, 1)
        axesHeat[i,j].set_title(type)
        if i == 1 and j == 2:
            sns.heatmap(plotVals[np.any(plotVals != 0, axis=1)], linewidth=0.5, cmap=col,center=0, vmin=0.8,vmax=2.8, xticklabels=times,yticklabels=[typeToName[type]+b for b in map(str,np.arange(0, len(units)).tolist())],cbar=False,ax=axesHeat[i,j])
        else:
           sns.heatmap(plotVals[np.any(plotVals != 0, axis=1)], linewidth=0.5, cmap=col,center=0, vmin=0.8,vmax=2.8, xticklabels=times,yticklabels=[typeToName[type]+b for b in map(str,np.arange(0, len(units)).tolist())],cbar=False,ax=axesHeat[i,j])

            #if heatMaps:
               # plt.savefig("pics/heatmap/"+type+"_heatmap_v3.png")
        #plt.close()
        stdFano = np.std(fanoAllMatched[typeToNum[type]], axis=0)
        #if shouldPlot:
            #for unit in range(0,5):
                #plt.plot(times, fanoAllMatched[typeToNum[type]][unit], color = FFcolor, linewidth = 1)
            #plt.plot(times, np.mean(fanoAllMatched[typeToNum[type]][0:len(fanoAllMatched[typeToNum[type]])-2],axis=0), color = FFcolor, linewidth = 2)
            #plt.show()
        print("Results of type: ", type)
        print("\tStd: ",stdFano)
        slopeList = minMax(fanoAllMatched,type, units, times, input_data)
        #ax = sns.heatmap(slopeList[:-1], linewidth=0.5, cmap=col, center=0, yticklabels=units[:-1])
        #if heatMaps:
           # plt.tight_layout()
            #plt.savefig("pics/slopePics/"+type+"_slope.png")
            #plt.close()

        if j > 1:
            i += 1
            j = 0
        else:
            j += 1

    print("Is done!")
    #figHeat.tight_layout(rect=[0, 0, .9, 1])
    figHeat.set_size_inches(16,9)
    figHeat.savefig("pics/heatmap/combined_heatmap_not_relative_v4.png")
    with open("postProcessedData/PostData2.json", "w") as write_file:
        json.dump(differentFromTheory, write_file, cls=NumpyArrayEncoder, indent=4,skipkeys=False)
    with open('postProcessedData/nameTranslations.txt', 'w') as filehandle:
        json.dump(neuronNameTranslationTable, filehandle)
    

def minMax(arr,type, typeUnits, times, input_data):
    len_tim = len(times)
    #print(typeUnits)
    maxPos = np.argmax(arr[typeToNum[type]], axis=1)
    minPos = np.argmin(arr[typeToNum[type]], axis=1)
    len_different = len(typeUnits)
    #print("\tMax\t: Min")
    diffData = {}
    numWrong = 0

    incDecSameList = np.empty((len_different,1))

    for i in range(0,len_different):
        #print("\t",maxPos[i],"\t: ", minPos[i])
        isWrong, typeError, slope = determine_trend(arr[typeToNum[type]][i])
        incDecSameList[i] = typeError
        #if (arr[typeToNum[type]][i][int(len_different/2)] > arr[typeToNum[type]][i][0] and arr[typeToNum[type]][i][int(len_different/2)] > arr[typeToNum[type]][i][1]) or (arr[typeToNum[type]][i][5] > arr[typeToNum[type]][i][0] and arr[typeToNum[type]][i][3] > arr[typeToNum[type]][i][0]):
        if isWrong:
            print("\t\t\tunit: ", typeToName[type]+str(i), " NO")
            ##print("\t\t\t\tSlope is: ", arr[typeToNum[type]][i][5] - arr[typeToNum[type]][i][0])
            ##print("\t\t\t\tDifference in Fano factor is: ", arr[typeToNum[type]][i][5] - arr[typeToNum[type]][i][0])
            print("\t\t\t\tRate is: ",np.mean(input_data[type][typeUnits[i]]['res']['meanRateSelect']))
            diffData[typeUnits[i]] = {"diff":arr[typeToNum[type]][i][5] - arr[typeToNum[type]][i][0], "rate":np.mean(input_data[type][typeUnits[i]]['res']['meanRateSelect']), "errorType":typeError}
            numWrong += 1
        else:
            print("\t\t\tunit: ", typeToName[type]+str(i), " YES")
            print("\t\t\t\tRate: ",np.mean(input_data[type][typeUnits[i]]['res']['meanRateSelect']))

    diffData["stat"] = {"numWrong":numWrong, "percWrong":numWrong/len_different}
    differentFromTheory[type] = diffData

    return incDecSameList


def determine_trend(values):
    # Slice the array to consider only the first half
    slope, _, _, _, _ = linregress(np.arange((len(values)//2)), values[1:(len(values)//2)+1])
    if slope > noDiffThreshold:
        '''Increase'''
        return True, 1, slope
    elif slope < -noDiffThreshold:
        '''Decrease'''
        return False, 0, slope
    else:
        '''No change'''
        return True, -1, slope

if __name__ == "__main__":
    start_parsing(False, False)
    print("Done")
    