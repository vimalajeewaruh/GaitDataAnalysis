close all; clear all; clc
addpath('./MatlabFunctions/')

%% Set parameters for plotting
lw = 2.5; set(0, 'DefaultAxesFontSize', 16);fs = 15;msize = 10;
seed = 10;
rng(seed, "twister")

%% Wavelet parameters 
 
J = 15; n = 2^J; % signal length
L = 1; % decompsition levels 

% wavelet filters 
family = 'Symmlet';
filt = MakeONFilter(family,4); 

pairs = nchoosek(1 :J-1, 2); 
a = 1; b = 12;
pairs = pairs(find( pairs(:,1) >= a & pairs(:,2 ) <=b ),:);

% ismean >> 0 - weigted median, 1 = weighted mean, 2 - Median, 3 - Mean 
ismean = 0;

H     = .20:.1:.8;%linspace(0.1,.7, 5);
nrep = 1000;

H_est = zeros(length(H), nrep); H_est_old= zeros(length(H), nrep);

for j = 1: length(H)
    for i = 1: nrep
        data = MakeFBMNew(2^J, H(j));
        
        wddata = dwtr(data, J - L, filt);
        
        h_hat = MomentMatchHurst_new(wddata, pairs, L, ismean);
        H_est(j, i) = h_hat;
    
        [slope, levels, log2spec] = waveletspectra_new(data, L, filt, a, b, 0, 0);
        H_est_old(j, i) = (slope * (-1) - 1)/2;
    
    end
end 


%%
h= figure('Renderer', 'painters', 'Position', [5 12 1200 600]);

trial1 = H_est_old';
trial2 = H_est';

% These grouping matrices label the columns:
grp1 = repmat(1:size(H_est_old,1),size(trial1,1),1);
grp2 = repmat(1:size(H_est,1),size(trial2,1),1);


% These color matrices label the matrix id:
clr1 = repmat(1,size(trial1));
clr2 = repmat(2,size(trial2));

% Combine the above matrices into one for x, y, and c:
x = [grp1;grp2];
y = [trial1;trial2];
c = [clr1;clr2];

% Convert those matrices to vectors:
x = x(:);
y = y(:);
c = c(:);

% Multiply x by 2 so that they're spread out:
x = x*2;

% Make the boxchart, 
boxchart(x(:),y(:),'GroupByColor',c(:))

% Set the x ticks and labels, and add a legend
xticks(2:2:14);
xticklabels(H)
legend('Standard', 'Proposed', 'NumColumns', 1,'Location','best')
xlabel('Actual Hurst Exponent'); ylabel('Estimated Hurst Exponent');% title('Case');
ylim([0.0 1.00]);
grid on 
%saveas(h,'./Figures/Test_Simulated_Standard_vs_New_Mean.png')



