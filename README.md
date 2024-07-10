Steps for using the codes 

# New Dimensions in Gait Dynamics Through an Advanced Self-similar Measure
This project proposes a new measure for characterizing self-similarity in high-frequency signals. The introduced method is applied in the context of an important gait data study. The steps below explain how to implement the measure and perform gait data analysis.

### Dataset
You can find the dataset available at https://zenodo.org/records/8003441; the dataset consists of linear acceleration (LA) and angular velocity (AV) measured across X, Y, and Z directions. These measurements were observed from 163 elderly participants, composed of 45 men and 118 women. Also, the demographic data file consists of several gait features. This project uses the raw LA and AV measurements to assess self-similarity and thirteen gait features.     

### Matlab Codes 
The repository includes Matlab files that are used to implement
  1. The new self-similarity measure,
  2. Simulation to assess performance of the new method,
  3. Gait data analysis involves implementing classifiers to assess discriminatory performance of the self-similar features computed using the new method. 

The **MatlabFunctions** folder contains a set of functions used in the Matlab files. In addition, the **wavelab 850** package available at https://github.com/gregfreeman/wavelab850 is required to run these codes.  

In the following, a brief introduction, for each code, is provided to explain its functionality.

### 1. Implementation of the new method

1. **MomentMatchHurst_new.m:** Implements the proposed Hurst exponent estimation method and is Available in the *MatlabFunctions* folder.
2. **waveletspectra_new.m:** Computes the Hurst exponent using the standard wavelet spectra-based method.

### 2.Simulation study
2. **New_H_vs_Starndard_H.m:** compares the Hurst exponent estimation performance between the standard and new method. This code mainly utilizes two functions *waveletspectra_new.m* and *MomentMatchHurst_new.m* available in **MatlabFunctions*. While the *waveletspectra_new.m* involves the Hurst exponent estimation using the standard wavelet spectra-based method, the *MomentMatchHurst_new.m* function computes the Hurst exponent using the proposed method in this project.
   
3. **Test_NewMethod_WaveFilters.m:** Compares the Hurst exponent estimation performance of the new and standard methods for different $H$ values with different wavelet filters and location measures.

### 3. Gait data analysis
1. **Test_DataProcessing.m:** Reads data from the data repository and separates cases and controls based on the falls and non-falls information provided in the GSTRIDE_database.csv. This code produces **Subject.mat** file containing cases and control IDs and **Gait_Ca.csv** and **Gait_Co.csv** containing LA and AV signals selected from the **Test_recording_raw** folder available in the data repository.

2. **Test_Gait_Features.m:** Reads 26 features from the **GSTRIDE_database** (*StrideTime AVG, StrideTime STD, Load Avg, Load STD, FootFlat AVG, FootFlat STD, Push AVG, Push STD, Swing AVG, Swing STD, Toe-Off Angle AVG, Toe-Off Angle STD, Heal Strike AVG, Heal Strike STD, Cadence AVG, Cadence STD, Step Speed AVG, Step Speed STD, Stride Length AVG, Stride Length STD*). The *AVG* and *STD* stand for average and standard deviation, respectively. This code computes their coefficient of variation and they are used as gait features in this project.

3. **Test_Slope_Features.m:** Computes the self-similar features of LA and AV signals using functions *waveletspectra_new.m* and *MomentMatchHurst_new.m*.
   
4. **Test_Significance_test.m:** Utilizes Wilcoxon rank sum tests to determine whether differences between non-fallers and fallers (gait and self-similarity) are statistically significant. 

5. **Test_Perfromance_Additional.m:** Implements classifiers to test discriminatory performance of the  gait features as well as self-similar features derived from the standard and new method. The classifiers include *Logistic regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, Naive Bayes, and Ensemble models. The main steps are as follows:
6. 
      i. Use the forward feature selection method to select the gait feature set that contributes to the higher classification performance.\
      ii. Perform classification with the selected gait features.\
      iii. As a basis for classification, feature matrices are constructed using selected gait features and all possible combinations of self-similar features. The combination of self-similar features that results in the best classification performance is used as the basis for the classification. This is performed with the self-similar features computed using the standard and new methods. 
 


