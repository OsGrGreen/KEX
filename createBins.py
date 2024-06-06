import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def binSpikes(session, params):
    time_step = params.get('step', 0.1)
    time_start = params.get('start', -0.1)
    time_end = params.get('end', 0.5)
    time_bins = np.arange(time_start, time_end + time_step, time_step) # Vi måste ha bättre sätt att sätta innan och efter.
    presentations = params.get('presentations')
    units = params.get('units')

    binnedSpikes = session.presentationwise_spike_counts( ## Skapa ett histogram där vi bara använder det valda stimulus, de valda units:en och vilka tider som ska användas
        stimulus_presentation_ids=presentations,  
        bin_edges=time_bins,
        unit_ids=units
    )

    presentationOrUnitWise = params.get('type', 0)

    if presentationOrUnitWise == 0:
        #print("Average over presentations")
        binnedSpikes = binnedSpikes.mean(dim="stimulus_presentation_id")
    else: 
        #print("Average over units")
        binnedSpikes = binnedSpikes.mean(dim="unit_id")
    
    return binnedSpikes




