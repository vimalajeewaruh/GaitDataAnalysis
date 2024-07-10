function [h_hat, E] = MomentMatchHurst_new(wddata, pairs, L, ismean)
    
    %% Inputs 
    % wdata - wavelet transformed data
    % pairs - scale paris used to estimate H
    % L - decomposition level  
    % ismean - H is estimated using weighted mean or weighted median

    %% Outputs
    % h_hat - estimated H
    % E  - Energy pairs used to estimated H (cross checking with the standard method)

    %% 

    %  log 2 of transformed signal length
    J = log2(size(wddata,2));

    H_hat = zeros(size(pairs, 1), 2);
    
    E = [];
    for p = 1: size(pairs,1)
        
       % select two arbitary levels 
        k1 = pairs(p,1); k2 = pairs(p,2); 

        l1 = k1; l2 = k2;

        % indexes to select wavelet coefficients 
        k1_indx =  2^(l1) + 1 : 2^(l1 + 1); 
        k2_indx =  2^(l2) + 1 : 2^(l2 + 1);
        
        % Log2 of median of squared wavelet coeffcients at level l1 and l2      
        d_k1 = log2( median ( wddata( k1_indx ).^2));  
        d_k2 = log2( median ( wddata( k2_indx ).^2));

        % Expected value of the hurst exponent using energis at level l1 and
        A = psi( 2^( l1 - 1 ) ) - psi( 2^(l2 - 1) ) ;
        B = log2( exp(1) ) * A - d_k1 + d_k2;
        C = l1 - l2;

        h_p = ( 1/ ( 2*C ) ) * ( B ) - 1; 
        
        % variance of the Hurst expoenent using energis at level l1 and
        % l2
        h_p_var = ( 1/ ( (2*C)^2 ) ) * ( psi( 1, 2^( l2 - 2 ) ) + psi( 1, 2^(l1 - 2) ) );
                
        H_hat(p, :) =  [ h_p sqrt(h_p_var) ];

        E = [E; [d_k1 d_k2]];
   end
    % Weights - reciporcal of variance of the  Hurst exponent        
    weights = ( 1./H_hat(:,2)) ./ sum(1./H_hat(:,2));
   
    % Estimate final Hurst exponent using all possible pairs of levels
    % between l1 and l2
    if ismean == 3
        h_hat = mean(H_hat(:,1)); % arithmetic median
    elseif ismean == 2 
        h_hat = median(H_hat(:,1)); % arithmetic mean 
    elseif  ismean == 1                
        h_hat =  sum(H_hat(:,1).* weights); % weighted mean
   elseif ismean == 0    
        h_hat = weighted_median(H_hat(:,1), weights); % weighted median
   end