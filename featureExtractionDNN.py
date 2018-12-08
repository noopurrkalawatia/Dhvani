import glob
import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


featureVectorLength = 140  # n_mfcc * number_of_summary_statistics

def extractFeaturesFromWaveform(file_name):
    rawData, sampleRate = librosa.load(file_name)

    # one row per extracted coefficient, one column per frame
    mfccs = librosa.feature.mfcc(y=rawData, sr=sampleRate, n_mfcc=20)

    mfccs_min = np.min(mfccs, axis=1)  # row-wise summaries
    mfccs_max = np.max(mfccs, axis=1)
    mfccs_median = np.median(mfccs, axis=1)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_variance = np.var(mfccs, axis=1)
    mfccs_skeweness = skew(mfccs, axis=1)
    mfccs_kurtosis = kurtosis(mfccs, axis=1)

    return mfccs_min, mfccs_max, mfccs_median, mfccs_mean, mfccs_variance, mfccs_skeweness, mfccs_kurtosis

def extract_features_from_directories(parentDirectory, sub_dirs, file_ext="*.wav"):
    feature_matrix, labels = np.empty((0, featureVectorLength)), np.empty(0)

    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parentDirectory, sub_dir, file_ext)):
            try:
                mfccs_min, mfccs_max, mfccs_median, mfccs_mean, mfccs_variance, mfccs_skeweness, mfccs_kurtosis = extractFeaturesFromWaveform(fn)
                print("Finished processing file: ", fn)
            except Exception as e:
                print("Error while processing file: ", fn)
                continue

            # concatenate extracted features
            new_feature_vector = np.hstack([mfccs_min, mfccs_max, mfccs_median, mfccs_mean, mfccs_variance, mfccs_skeweness, mfccs_kurtosis])

            # add current feature vector as last row in feature matrix
            feature_matrix = np.vstack([feature_matrix, new_feature_vector])

            # extracts label from the file name. Change '\\' to  '/' on Unix systems
            labels = np.append(labels, fn.split('/')[2].split('-')[1])

    return np.array(feature_matrix), np.array(labels, dtype=np.int)

def prepareFeatures(trainingDirectory, validationDirectory, trainingName, validationName):

    parentDirectory = 'Sound'  # name of the directory which contains the recordings
    trainingSubDirectories = trainingDirectory
    validationSubDirectories = validationDirectory

    # ndarrays
    training_features, training_labels = extract_features_from_directories(parentDirectory, trainingSubDirectories)
    test_features, test_labels = extract_features_from_directories(parentDirectory, validationSubDirectories)

    # convert ndarray to pandas dataframe
    training_examples = pd.DataFrame(training_features, columns=list(range(1, featureVectorLength+1)))
    # convert ndarray to pandas series
    training_labels = pd.Series(training_labels.tolist())

    # convert ndarray to pandas dataframe
    validation_examples = pd.DataFrame(test_features, columns=list(range(1, featureVectorLength+1)))
    # convert ndarray to pandas series
    validation_labels = pd.Series(test_labels.tolist())

    # store extracted training data
    training_examples.to_pickle('Extracted_Features-' + trainingName + '_features.pkl')
    training_labels.to_pickle('Extracted_Features-' + trainingName + '_labels.pkl')

    # store extracted validation data
    validation_examples.to_pickle('Extracted_Features-' + validationName + '_features.pkl')
    validation_labels.to_pickle('Extracted_Features-' + validationName + '_labels.pkl')


# First 9 folds will be used for training, the tenth for validation.
trainingDirectory = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9"]
validationDirectory = ["fold10"]
prepareFeatures(trainingDirectory, validationDirectory, 'notFold10', 'fold10')

# Read the stored features and labels:
print(pd.read_pickle('Extracted_Features-fold10_features.pkl')) 
print(pd.read_pickle('Extracted_Features-fold10_labels.pkl'))