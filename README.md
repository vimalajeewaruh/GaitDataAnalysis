Steps for using the codes 

# New Dimensions in Gait Dynamics Through an Advanced Self-similar Measure
This project proposes a new measure for characterizing self-similarity in high-frequency signals. The introduced method is applied in the context of an important gait data study. The steps below explain how to implement the measure and perform gait data analysis.

### Dataset
You can find the dataset available at https://zenodo.org/records/8003441; the dataset consists of linear acceleration (LA) and angular velocity (AV) measured across X, Y, and Z directions. These measurements were observed from 163 elderly participants, composed of 45 men and 118 women. Also, the demographic data file consists of several gait features. This project uses the raw LA and AV measurements to assess self-similarity and thirteen gait features.     

### Matlab Codes 
The repository includes Matlab files that are used to implement
  1. The new self-similarity measure,
  2. Simulation to assess performance of the new method,
  3. classifiers to assess discriminatory performance of the self-similar features computed using the new method. 

The **MatlabFunctions** folder contains a set of functions used in the Matlab files. To run these codes, follow the instructions provided.

1. **Test_DataProcessing:** Reads data from the data repository and separates cases and controls based on the falls and non-falls information provided in the GSTRIDE_database.csv. This code produces **Subject.mat** file containing cases and control IDs and **Gait_Ca.csv** and **Gait_Co.csv** containing LA and AV signals selected from the **Test_recording_raw** folder available in the data repository.

2. **Test_Gait_Features:** Reads 26 features from the **GSTRIDE_database** ("StrideTime AVG", "StrideTime STD", "Load Avg", "Load STD", "FootFlat AVG","FootFlat STD", "Push AVG", "Push STD","Swing AVG", "Swing STD", "Toe-Off Angle AVG","Toe-Off Angle STD", "Heal Strike AVG", "Heal Strike STD", "Cadence AVG", "Cadence STD","Step Speed AVG",  "Step Speed STD", "Stride Length AVG", "Stride Length STD") and they quantify the average and standard deviation of thirteen gait features. This code computes their coefficient of variation and they are used as gait features in this project. 


1.  The study utilized VGRF data files, which were extracted from the Physionet repository. The **CaseFeatures** and **ControlFeatures** folders contain multiscale and time-domain features that were generated using the VGRF data.

   **Multi-scale features**\
      i. Level-wise cross-correlation \
      ii. Wavelet entropy\
      iii. Spectral slope 
    
   **Time-domain features** \
      i. Stance time and Swing Time\
      ii. Maximum force reaction at toe off

2.  Run **Demo** files to test the classification performance of multiscale features and their integration with three time-domain features in diagnosing Parkinson's disease

