import matplotlib.pyplot as plt
import numpy as np

def newPlotFano(Result, display, name):
    sep = 0.15
    relGain = 2
    FFcolor = [0]*3
    MNcolor = [0]*3
    MNallColor = [0.7]*3
    FFscale = []
    MNscale = []
    FFoffset = []
    MNoffset = []

    pad = 0.025

    maxFF = [max(i) for i in zip(*Result.get('Fano_95CIs'))][1]
    maxFF = maxFF + pad

    minFF = [min(i) for i in zip(*Result.get('Fano_95CIs'))][0]
    minFF = minFF - pad

    maxMN = max(max(Result.get('meanRateAll')), max(Result.get('meanRateSelect')))
    minMN = min(min(Result.get('meanRateAll')), min(Result.get('meanRateSelect')))

    topFFportion = (1 - sep)*(relGain/(1 + relGain))
    bottomMNportion = topFFportion + sep

    rngFF = max(0.3, maxFF - minFF)
    FFscale = topFFportion/rngFF
    FFoffset = topFFportion/2 - FFscale*(minFF + maxFF)/2

    rngMN = max(5, maxMN - minMN)
    MNscale = (1 - bottomMNportion) / rngMN
    MNoffset = bottomMNportion - MNscale*minMN

    plt.figure()
    plt.title("Fano factor for "+name)
    plt.ylabel("Fano factor")
    plt.xlabel("Times")
    plt.plot(Result.get('times'), Result.get('FanoFactor'), color = FFcolor, linewidth = 2)
    plt.plot(Result.get('times'), Result.get('Fano_95CIs'), color = FFcolor, linewidth = 1)
    plt.vlines(x=0, ymin=min(np.min(Result.get('FanoFactor')), np.min(Result.get('Fano_95CIs'))), ymax=max(np.max(Result.get('FanoFactor')), np.max(Result.get('Fano_95CIs'))), colors='darkgrey', ls=':', lw=2)
    if not display:
        plt.savefig("pics/"+name+"_fano.png")
    plt.close()
    plt.figure()
    plt.title("Firing rate for "+name)
    plt.ylabel("Firing rate")
    plt.xlabel("Times")
    plt.plot(Result.get('times'), Result.get('meanRateAll'), color = MNallColor, linewidth = 2)
    plt.plot(Result.get('times'), Result.get('meanRateSelect'), color = MNcolor, linewidth = 2)
    if display:
        plt.show()
    else:
        plt.savefig("pics/"+name+"_rate.png")
    plt.close()