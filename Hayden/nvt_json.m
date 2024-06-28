
clc;
clear all;

addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Matlab framework\framework\framework\IO')
addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Matlab framework\framework\framework\neuralynx')
addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Matlab framework\framework\framework\behav')
addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Matlab framework\NeuralynxMatlabImportExport_v413')
addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Matlab framework\framework')
addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Matlab framework\framework\framework\contrib')
addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Matlab framework\framework\segmentation')
addpath('C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Tmaze_Data\252-1375\2018-01-07_15-14-54\04_tmaze1')



savepath;
% Define the input file path
nvtFilePath = 'VT1_fixed.nvt';  % Update this path to your actual .nvt file location

% Define the output JSON file path
jsonFilePath = 'C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Tmaze_Data\252-1375\2018-01-07_15-14-54\04_tmaze1\nvt_json\nvt_data.json';


PARAM = struct();
PARAM.pix2cm = 5.4135;
PARAM.nvtFilePath = 'C:\Users\huangjuhua\Desktop\Wisniewski and Masurkar Labs\MatlabFramework_SampleCode_Data\Tmaze_Data\252-1375\2018-01-07_15-14-54\04_tmaze1\VT1_fixed.nvt';


% Load the .nvt data
[TSusec, Xpos1, Ypos1, Ang1] = NlxNvtLoadYflip(PARAM.nvtFilePath, PARAM.pix2cm);
%[TSusec, Xpos1, Ypos1, Ang1] = NlxNvtLoadYflip(nvtFilePath, pix2cm);

% Structure the data into a MATLAB struct
dataStruct = struct(...
    'TSusec', TSusec, ...
    'Xpos1', Xpos1, ...
    'Ypos1', Ypos1, ...
    'Ang1', Ang1 ...
);

% Convert the structured data to JSON format
jsonData = jsonencode(dataStruct);

% Write JSON data to file
fileId = fopen(jsonFilePath, 'w');
if fileId == -1
    error('Failed to open file for writing JSON data.');
end
fprintf(fileId, '%s', jsonData);
fclose(fileId);

% Confirm the process is completed
disp('JSON data has been saved successfully.');