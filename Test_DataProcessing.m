close all; clear all; clc
addpath('/Users/hvimalajeewa2/Documents/Tamu_Documents/TAMU/DemosNew/')
%addpath('/Users/dixon/Documents/TAMU/Sample Data/White blood Cancer data/MatlabFunctions/')
%addpath('/Users/hvimalajeewa2/Documents/UNL_Documents/Projects/Gait_Data/GSTRIDE_database/Test_recordings_raw/')

%%  ############################ Data Label Files ##############################

dirName = sprintf('/Users/hvimalajeewa2/Documents/UNL_Documents/Projects/Gait_Data/GSTRIDE_database/');             %# folder path
files = dir( fullfile(dirName,'*.csv') );

fname = fullfile(dirName,files.name);     %# full path to file
data_registry = readtable(fname, 'VariableNamingRule','preserve');
data_label = table2array(data_registry(:,5)); 
data_device = table2array(data_registry(:,6));
data_environment = table2array(data_registry(:,11));

%% ##############################  Raw Data Files ####################

dirName = sprintf('/Users/hvimalajeewa2/Documents/UNL_Documents/Projects/Gait_Data/GSTRIDE_database/Test_recordings_raw/');             %# folder path
files = dir( fullfile(dirName,'*.txt') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);

%% ##############################  Seperate Case and Control ####################
Case_ID = []; Control_ID = [];
for i = 1:nfi
    dataFile_Name = cell2mat(files(i));
     newStr = split(dataFile_Name,'_');
    
     if ismember(data_device(i+1), 'CSIC') %&& ismember(data_environment(i+1), 'M')
         if ismember(data_label(i+1), 'NO')  
            Control_ID =  [Control_ID; i];
         else 
            Case_ID =  [Case_ID; i ];
         end
     end 
end 

%% Save Case and Cotrol raw data files in Case and Control Folders
% Case

Case_ID_new = [];

d = './Case_csv/'; 
for i =  1:length(Case_ID)
    id = Case_ID(i);
    fname = fullfile(dirName,files{id});     %# full path to file
    data =  readtable(fname, 'VariableNamingRule','preserve');
    if ~isempty(data) && size(data,1) > 1000
        dataFile_Name = cell2mat(files(id));
        newStr = split(dataFile_Name,'_');
        
        filename = sprintf('CaseData_%s.csv',cell2mat(newStr(1)));
        writetable(data,fullfile(d,filename))
        Case_ID_new = [Case_ID_new id];
    end 
end 

% Control 
Control_ID_new = [];

d = './Control_csv/'; 
for i =  1:length(Control_ID)
    id = Control_ID(i);
    fname = fullfile(dirName,files{id});     %# full path to file
    data =  readtable(fname, 'VariableNamingRule','preserve');
    if ~isempty(data) && size(data,1) > 1000
        dataFile_Name = cell2mat(files(id));
        newStr = split(dataFile_Name,'_');
        
        filename = sprintf('ControlData_%s.csv',cell2mat(newStr(1)));
        writetable(data,fullfile(d,filename));

        Control_ID_new = [Control_ID_new id];
    end 
end

Subject.Control_ID_new = Control_ID_new;
Subject.Case_ID_new = Case_ID_new;
save( 'Subject.mat' )
%% Clibrated descriptors

dirName = sprintf('/Users/hvimalajeewa2/Documents/UNL_Documents/Projects/Gait_Data/GSTRIDE_database/');             %# folder path
files = dir( fullfile(dirName,'*.csv') );

fname = fullfile(dirName,files.name);     %# full path to file
data_registry = readtable(fname, 'VariableNamingRule','preserve');


d = './Ex_Features/'; 
Gait_features_Ca = data_registry(Case_ID_new,:);
filename = 'Gait_Ca.csv';
writetable(Gait_features_Ca,fullfile(d,filename));

Gait_features_Co = data_registry( Control_ID_new,:);
filename = 'Gait_Co.csv';
writetable(Gait_features_Co,fullfile(d,filename));



