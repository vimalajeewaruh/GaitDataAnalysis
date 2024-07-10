close all; clear all; clc
addpath('./MatlabFunctions/')

%% INITIALIZING PARAMETERS 
% measure used to compute the standard wavelet spectra: 1 - mean, 0 - median 
ismean_o = 0;
% measure used to compute final H in new method; options 0- weighted median, 1 - weighted mean
%  2 - arithmetic median, 3 - arithmetic mean 
ismean_n = 0; 

% plot wavelet spectra 0 - No/ 1 - yes
isplot = 1;          

% Linear acceleration and angular velocity 
Ss = ["LA_X","LA_Y", "LA_Z", "AV_X",  "AV_Y",  "AV_Z" ];

% signal length used to perform wavelet transform
n = 2^15; 
% Number of datapoints to omit if they are zero
d = 500;

% Wavelet decompositions  J - L, where J = log2(n)
L = 1;

% Wavelet filter
family = 'Symmlet'; filt = MakeONFilter(family,8); 

% The range of levels used to estimate H 
k11 = 8; k12 = 12; %  scales use to estimate H of LA signals
k21 = 8; k22 = 12; %  scales use to estimate H of AV signals

% The second range of levels shows the second slope. 
% k21 = 2; k22 = 6;
% k11 = 2; k12 = 6;

%%  ############################  Estimate H usign the standard wavelet spectra method for cases ##############################
% load raw LA and AV signals from people at risk of falls
dirName = sprintf('./Case_csv/');             %# folder path

files = dir( fullfile(dirName,'*.csv') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);

% Estimate H usign the standard wavelet spectra method for cases
Slope_Case_Old = zeros(nfi, 6);  KK2 = [];

h= figure('Renderer', 'painters', 'Position', [5 10 1200 600]);

for i = 1: nfi
    %id = Case_ID(i);
    fname = fullfile(dirName,files{i});     %# full path to file
    data =  readtable(fname, 'VariableNamingRule','preserve');
    data_csv = table2array(data); 
    
    J = floor(log2(size(data_csv,1)));  
    
    for j = 1: size(data_csv,2)
        data = data_csv(:,j);
        
        if length(data) > n
            data = data( d:end);
        end 
        data = data(1:n);

        subplot(2,3,j)

        if j < 4
            [slope, levels, log2spec] = waveletspectra_new(data, L, filt, k11, k12, ismean_o, isplot);
        end 
        if j >= 4
            [slope, levels, log2spec] = waveletspectra_new(data, L, filt, k21, k22, ismean_o, isplot); 
        end 

        Slope_Case_Old(i,j) = ( -slope -1)/2;
        grid on
        title(Ss(j))
    end     
    
end 
saveas(h,'./Figures/Test_Standard_Case.png')

%% Estimate H usign the standard wavelet spectra method for controls
% load raw LA and AV signals from people at risk of falls
dirName = sprintf('./Control_csv/');             %# folder path

files = dir( fullfile(dirName,'*.csv') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);

Slope_Control_Old = zeros(nfi, 6);

h= figure('Renderer', 'painters', 'Position', [5 10 1200 600]);

for i = 1: nfi
    %id = Case_ID(i);
    fname = fullfile(dirName,files{i});     %# full path to file
    data =  readtable(fname, 'VariableNamingRule','preserve');
    data_csv = table2array(data); 

    J = floor(log2(size(data_csv,1))); L = 1; 
    KK2 = [KK2 J];

    Slope = zeros(nfi, size(data_csv,2));
    for j = 1: size(data_csv,2)
        
        data = data_csv(:,j);
        
        if length(data) > n
            data = data( d:end);
        end 
        data = data(1:n);

        subplot(2,3,j)
        if j < 4
            %k11 = 8; k12 = 14;
            [slope, levels, log2spec] = waveletspectra_new(data, L, filt, k11, k12, ismean_o, isplot);
        end 

        if j >= 4
            %k21 = 10; k22 = 14;
            [slope, levels, log2spec] = waveletspectra_new(data, L, filt, k21, k22, ismean_o, isplot);
        end 
        Slope_Control_Old(i,j) = ( -slope -1)/2;
        grid on 
        title(Ss(j))
    end       
end 
saveas(h,'./Figures/Test_Standard_Control.png')


%% %%%%%%%%%%%%%%%%%%%%%%% New Hust exponent measure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load raw LA and AV signals from people not at risk of falls
dirName = sprintf('./Control_csv/');             %# folder path

files = dir( fullfile(dirName,'*.csv') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);

Slope_Control_New = zeros(nfi, 6);

for i = 1: nfi

    fname = fullfile(dirName,files{i});     %# full path to file
    data =  readtable(fname, 'VariableNamingRule','preserve');
    data_csv = table2array(data); 
    
    J = floor(log2(size(data_csv,1)));  J = 15;
    pairs = nchoosek(1 :J-1, 2); 

    Slope = zeros(nfi, size(data_csv,2));
    for j = 1: size(data_csv,2)
        
        % load LA and AV signals 
        data = data_csv(:,j);
        % Ingnore first d measurements if they are zero
        if length(data) > n
            data = data( d:end);
        end 
        data = data(1:n);
        
        % Perform wavelet transforms
        wddata = dwtr(data, J - L, filt);
        
        % Estimate H of LAs
        if j < 4
            pairs1 = pairs(find( pairs(:,1) >= k11 & pairs(:,2 ) <= k12 ),:);
            h_hat =  MomentMatchHurst_new(wddata, pairs1,L, ismean_n);
        end

        % Estimate H of AVs
        if j >= 4 
           pairs2 = pairs(find( pairs(:,1) >= k21 & pairs(:,2 ) <= k22 ),:);
           h_hat =  MomentMatchHurst_new(wddata, pairs2,L, ismean_n);
        end

        Slope_Control_New(i,j) = h_hat;

    end       
end 


%% load raw LA and AV signals from people at risk of falls

dirName = sprintf('./Case_csv/');             %# folder path

files = dir( fullfile(dirName,'*.csv') );   %# list all *.xyz files
files = {files.name}';                      %'# file names
nfi = numel(files);


Slope_Case_New = zeros(nfi, 6);

for i = 1: nfi
    %id = Case_ID(i);
    fname = fullfile(dirName,files{i});     %# full path to file
    data =  readtable(fname, 'VariableNamingRule','preserve');
    data_csv = table2array(data); 
    
    J = floor(log2(size(data_csv,1))); L = 1; J = 15;
    pairs = nchoosek(1 :J-1, 2); 

    Slope = zeros(nfi, size(data_csv,2));

    for j = 1: size(data_csv,2)
        
        % select LA and AV signals from the raw data 
        data = data_csv(:,j);
        
        % ignore if there are zeros in the begenning of the signal
        if length(data) > n
            data = data( d:end);
        end 
        data = data(1:n);
        
        % Perform wavelet transforms on data
        wddata = dwtr(data, J - L, filt);

       if j < 4
            pairs1 = pairs(find( pairs(:,1) >= k11 & pairs(:,2 ) <= k12 ),:);
            h_hat =  MomentMatchHurst_new(wddata, pairs1,L, ismean_n);
        end
        if j >= 4 
           pairs2 = pairs(find( pairs(:,1) >= k21 & pairs(:,2 ) <= k22 ),:);
           h_hat =  MomentMatchHurst_new(wddata, pairs2,L, ismean_n);
        end
        Slope_Case_New(i,j) = h_hat;

    end       
end 

%% Plot H of LA and AV from the old and new method for cases and controls

h= figure('Renderer', 'painters', 'Position', [5 10 1200 600]);

trial1 = Slope_Case_Old;%rand(5,7);
trial2 = Slope_Control_Old;%rand(10,7);
trial3 = Slope_Case_New;%rand(5,7);
trial4 = Slope_Control_New;%rand(10,7);

% These grouping matrices label the columns:
grp1 = repmat(1:size(Slope_Case_Old,2),size(trial1,1),1);
grp2 = repmat(1:size(Slope_Case_Old,2),size(trial2,1),1);
grp3 = repmat(1:size(Slope_Control_Old,2),size(trial3,1),1);
grp4 = repmat(1:size(Slope_Control_Old,2),size(trial4,1),1);

% These color matrices label the matrix id:
clr1 = repmat(1,size(trial1));
clr2 = repmat(2,size(trial2));
clr3 = repmat(3,size(trial3));
clr4 = repmat(4,size(trial4));


% Combine the above matrices into one for x, y, and c:
x = [grp1;grp2;grp3;grp4];
y = [trial1;trial2;trial3;trial4];
c = [clr1;clr2;clr3;clr4];

% Convert those matrices to vectors:
x = x(:);
y = y(:);
c = c(:);

% Multiply x by 2 so that they're spread out:
x = x*2;

% Make the boxchart, 
boxchart(x(:),y(:),'GroupByColor',c(:))

% Set the x ticks and labels, and add a legend
xticks(2:2:12);
xticklabels(["LA_X","LA_Y", "LA_Z", "AV_X",  "AV_Y",  "AV_Z" ])
legend('Fallers_{Standard}', 'Non-Fallers_{Standard}', 'Fallers_{New}', 'Non-Fallers_{New}', 'NumColumns', 2,'Location','best')
xlabel('Sensor (LA - Linear Acceleration, AV - Angular Velocity)'); ylabel('Hurst Exponent (H)');% title('Case');
ylim([-0.50 1.50]);
grid on 

%saveas(h,'./Figures/Test_Standard_vs_New_H_2_6.png') % plot for energies within scales 2 and 6

%saveas(h,'./Figures/Test_Standard_vs_New_H_8_12.png') % plot for energies within scales 8 and 12

%% Test the differnece between H of LA and AV from cases and controls is significant 

disp("Test the differnece between H of LA and AV from cases and controls from the standard method")

% cases
X_control_Slp = Slope_Control_Old; 
X_case_slp = Slope_Case_Old; 

H = [];
for i = 1:size(X_case_slp,2)
    A = X_control_Slp(:,i); B = X_case_slp(:,i);
    
    % Wilcoxon rank sum test
    [p1, h, stats] = ranksum(A, B);
    
    H = [H h];

    if h == 0
        disp('The test fails to reject the null hypothesis. There is no significant difference between the two samples.');
    else
        disp('The test rejects the null hypothesis. There is a significant difference between the two samples.');
    end
end 

% Select significant factors 
Slope_Control_Old = Slope_Control_Old(:,find(H));
Slope_Case_Old = Slope_Case_Old(:,find(H));

% Controls

disp ("Test the differnece between H of LA and AV from cases and controls from the new method")

X_control_Slp = Slope_Control_New; 
X_case_slp = Slope_Case_New; 

H = [];
for i = 1:size(X_case_slp,2)
    A = X_control_Slp(:,i); B = X_case_slp(:,i);

    [p1, h, stats] = ranksum(A, B);
    H = [H h];
    if h == 0
        disp('The test fails to reject the null hypothesis. There is no significant difference between the two samples.');
    else
        disp('The test rejects the null hypothesis. There is a significant difference between the two samples.');
    end
end 

Slope_Control_New = Slope_Control_New(:,find(H));
Slope_Case_New = Slope_Case_New(:,find(H));

%% Save the H feature matrices for classfications

Y_Ca = ones(size(Slope_Case_New,1),1); Y_Co = zeros(size(Slope_Control_New,1),1);

GaitFeaturesMatrix.Slope_Case_New = Slope_Case_New;
GaitFeaturesMatrix.Slope_Control_New = Slope_Control_New;


GaitFeaturesMatrix.Slope_Case_old = Slope_Case_Old;
GaitFeaturesMatrix.Slope_Control_old = Slope_Control_Old;


GaitFeaturesMatrix.Y_Co = Y_Co;
GaitFeaturesMatrix.Y_Ca = Y_Ca;

%save('GaitFeaturesMatrix.mat')
