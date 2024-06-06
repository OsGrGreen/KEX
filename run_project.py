import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.visualization import plot_mean_waveforms, plot_spike_counts, raster_plot
from createBins import binSpikes
from VarVsMean import VarVsMean, scatterData, distData
from newPlotFano import newPlotFano
from json import JSONEncoder
import json

run = 0

def convert_keys_to_json_serializable(dictionary):
    converted_dict = {}
    for key, value in dictionary.items():
        if isinstance(key, np.int64):
            key = int(key)  # Convert int64 key to regular int
        elif isinstance(key, np.ndarray):
            key = key.tolist()  # Convert numpy array key to list
        if isinstance(value, dict):
            value = convert_keys_to_json_serializable(value)
        converted_dict[key] = value
    return converted_dict

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        print("Encoding: ", obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, scatterData):
            return ""
        elif isinstance(obj, distData):
            return ""
        return json.JSONEncoder.default(self, obj)


def single_neuron(unit, presentations, time_step,session):
    data = np.empty((0, 60, 350))
    counter = 0
    tempData = np.empty((60, 350))
    for val in presentations.index.values:
        #print("Is at presentation:", val)
        presentation = presentations[presentations.index.values == val]
        #for unit in units.index.values[0]:
        
        #print(units.index.values)
        # Fix so we only have one spike for each..
        params = {"step":time_step, "presentations": presentation.index.values, "units":unit, "end":0.25}
        binnedSpikes = binSpikes(session,params)
        #d = 0.5 - (-0.1)
        #s = np.sum(binnedSpikes)
        if counter == 60:
            counter = 0
            #print(tempData.shape)
            data = np.concatenate((data, tempData[np.newaxis, :, :]), axis=0)
            #tempData = np.empty((60, 600))
        else:
            tempData[counter] = binnedSpikes.T
            counter += 1
    data[data > 0.0] = 1.0
    return data

def single_type(session, presentations, unitType, output):
    
    units = session.units[session.units["ecephys_structure_acronym"] == unitType] ##Definiera vilka units vi vill använda/söka efter
    time_step = 0.001 
    times = np.array((range(100, 325, 25)))
    params = {"matchReps":5, "boxWidth":80, "alignTime":100}
    runUnits = units.index.values[0:1]
    runUnits = np.concatenate((runUnits,np.random.choice(units.index.values,2)))
    currentUnit = 0
    outputType = {}
    allData = np.empty((0, 60, 350))
    for unit in runUnits:
        print("\tIs at unit:", currentUnit, " / ", (len(runUnits)-1))
        print("\t\t Starting data extraction...")
        data = single_neuron(unit,presentations,time_step,session)
        allData = np.concatenate((allData, data[:, :, :]), axis=0)
        print("\t\t Starting calculations...")
        res = VarVsMean(data, times, params)
        if res:
            newPlotFano(res,False,unitType+str(unit))
            new_data = convert_keys_to_json_serializable(res)
            outputType[int(unit)] = {"unit":int(unit), "res":new_data}
        currentUnit += 1

    print("\tRunning for all units")
    print("\t\t Starting data extraction...")
    res = VarVsMean(allData, times, params)
    newPlotFano(res,False,unitType+"multi")
    new_data = convert_keys_to_json_serializable(res)
    outputType["multi"] = {"unit":runUnits, "res":new_data}

    output[unitType] = outputType
        
    

def start_measurement():
    output_dir = '/local1/ecephys_cache_dir/'

    manifest_path = os.path.join(output_dir, "manifest.json")

    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    sessions = cache.get_session_table()

    genoType = sessions.full_genotype.value_counts().index.values
    geno_session = sessions[sessions.full_genotype.str.match(genoType[0])]

    session_types = ["brain_observatory_1.1", "functional_connectivity"]
    use_session = geno_session[geno_session["session_type"] == session_types[0]]
    print()
    print(use_session.index.values[0])
    print()
    print()
    session = cache.get_session_data(use_session.index.values[0])

    
    units = session.structurewise_unit_counts.index.values
    
    presentations = session.get_stimulus_table("natural_scenes")

    print()
    print(presentations)
    print()

    val = 49433
    #presentation = presentations[presentations.index.values == val].index.values

    try:
        with open("processedData/pross_data2.json", "r") as read_file:
            output_data = json.load(read_file)
    except FileNotFoundError:
        output_data = {}

    for unitType in units:
        print("Is at unit type:", unitType)
        single_type(session,presentations,unitType, output_data)

    with open("processedData/pross_data2.json", "w") as write_file:
        json.dump(output_data, write_file, cls=NumpyArrayEncoder, indent=4,skipkeys=False)
    

if __name__ == "__main__":
    start_measurement()
    print("Done")
    