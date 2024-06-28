# ntt_json.ipynb
This a python code to convert .ntt files to json files. 

json Structure: 
['CellNumber', 'Params', 'Timestamps']

{"CellNumber":, 
  "Params":, 
   "Timestamps" : [Data1, Data2, ..., Data]
}


# ncs_json.ipynb

This a python code to convert .ncs files to json files. 

json Structure: 
['CellNumber', 'SampleFrequency', 'Number_of_valid_samples', 'Time_data']

'''
{"CellNumber":,
"SampleFrequency":,
  "Number_of_valid_samples":, 
   "Time_data" : {[Timestamp, Samples],[Timestamp, Samples],...,[Timestamp, Samples]}
}
'''


# nvt_json.ipynb
This a MatLab code to convert .nvt files to json files. You need to run in the matlab.

json Structure: 
['TSusec', 'Xpos1', 'Ypos1', 'Ang1']

'''
{"TSusec",
 "Xpos1", 
 "Ypos1", 
 "Ang1"}
'''
