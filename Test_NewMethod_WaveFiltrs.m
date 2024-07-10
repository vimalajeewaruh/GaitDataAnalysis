%close all; clear all; clc
addpath('./MatlabFunctions/')

%% Set parameters for plotting
lw = 2.5; set(0, 'DefaultAxesFontSize', 16);fs = 15;msize = 10;
seed = 10;
rng(seed, "twister")

%% Wavelet parameters 
 
J = 13; n = 2^J; % signal length
L = 1; % decompsition levels 

% wavelet filters
family = 'Symmlet';
filt = MakeONFilter(family,4); 
display(family)

% All possible pairs to be considered among decompsition levels 
pairs = nchoosek(1 :J-1, 2); 
a = 1; b = 12;
pairs = pairs(find( pairs(:,1) >= a & pairs(:,2 ) <=b ),:);

% Number of repetitions 
nrep = 1000;

% True hurst exponent values 
H = .2:.2:.8;

H_new = zeros(2, 4); H_est_old = zeros(2, 4);

% ismean >> 0 - Wmedian, 1 = Wmean
for k = [0, 1]  
    
    ismean = k;
    H_MSE_new = zeros(length(H), nrep); H_MSE_old= zeros(length(H), nrep);
    
    for j = 1: length(H)
        for i = 1: nrep
            % Generate a fractional brownian motion with known H
            data = MakeFBMNew(2^J, H(j)); 
            % Wavelet transforms 
            wddata = dwtr(data, J - L, filt);
            
            % Estimate H using the new method
            h_hat = MomentMatchHurst_new(wddata, pairs, L, ismean);
            H_MSE_new(j, i) = (h_hat - H(j))^2;
            
            % Estimated H using the standard method
            [slope, levels, log2spec] = waveletspectra_new(data, L, filt, a, b, 1, 0);
            h_hat = (slope * (-1) - 1)/2;
            H_MSE_old(j, i) = (h_hat - H(j))^2 ;
        
        end
    end 
  % Standard deviation of estimations   
  H_est_new(k+1, :) =  mean(H_MSE_new');
  % Mean of restimations 
  H_est_old(k+1, :) = mean(H_MSE_old, 2);
end 
H_est_old

H_est_new




