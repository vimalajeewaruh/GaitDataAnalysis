close all; clear all; clc

%%
dirName = sprintf('./Gait_Data/GSTRIDE_database/');             %# folder path
files = dir( fullfile(dirName,'*.csv') );

fname = fullfile(dirName,files.name);     %# full path to file
data_registry = readtable(fname, 'VariableNamingRule','preserve', 'DecimalSeparator','.');

load Subject.mat
Case_ID_new = Subject.Case_ID_new;
Control_ID_new = Subject.Control_ID_new;

Gait_Ca = data_registry(Case_ID_new,:);
Gait_Co = data_registry( Control_ID_new,:);
%%
Tests = ["Stride Time", "Load", "Foot Flat","Push", "Swing ", "Toe-Off Angle ",...
    "Heal Strike", "Cadence", "Step Speed", "Stride Length", "3D Path", "2D Path", "Clearance"] ;

h= figure('Renderer', 'painters', 'Position', [6 6 1200 1200]);
p = 35;
for i = 1:13
    AVG_ca = table2array(Gait_Ca(:,p));
    STD_ca = table2array(Gait_Ca(:,p+1));
    
    AVG_co = table2array(Gait_Co(:,p));
    STD_co = table2array(Gait_Co(:,p+1));

    A = STD_ca./AVG_ca;
    B = STD_co./AVG_co;

    Gait_Fe_Ca(:,i+4) = A(:,1); Gait_Fe_Co(:,i+4) = B(:,1);

    subplot(3,5,i)
    grp = [zeros(1,size(A,1)),ones(1,size(B,1))];
    boxplot([B' A'],grp, 'Labels',{'Non-fallers','Fallers'})
    title(Tests(i))
    grid on 
    p = p + 2;
end 

%saveas(h,'./Figures/Test_Gait_Features_CV.png')

%% Test the differnece between H of LA and AV from cases and controls is significant
disp("Test the differnece between H of LA and AV from cases and controls from the new method")

% Controls
X_control_Slp = Gait_Fe_Co; 
X_case_slp = Gait_Fe_Ca; 

H = [];
for i = 1:size(X_case_slp,2)
    A = X_control_Slp(:,i); B = X_case_slp(:,i);

    [p1, h, stats] = ranksum(A, B);

    if h == 0
        disp('The test fails to reject the null hypothesis. There is no significant difference between the two samples.');
    else
        disp('The test rejects the null hypothesis. There is a significant difference between the two samples.');
    end
end 
