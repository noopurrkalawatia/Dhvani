import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


featureVectorSize = 140
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


##Method name : computeFeatureColumns
##parama      : void
##return      : set of feature columns. 
def computeFeatureColumns():
    """Construct the TensorFlow Feature Columns.
    Returns:
      A set of feature columns
    """
    return set([tf.feature_column.numeric_column('audioFeatures', shape=featureVectorSize)])

##Method name : computeTrainInputFn
##parama      : features, labels, batch_size, num_epochs=None, shuffle=True
##return      : A function that returns batches of training features and labels during training.
def computeTrainInputFn(features, labels, batch_size, num_epochs=None, shuffle=True):
    def inputFunction(num_epochs=num_epochs, shuffle=True):
        random_array = np.random.permutation(features.index)
        raw_features = {"audioFeatures": features.reindex(random_array)}
        raw_labels = np.array(labels[random_array])

        datatens = Dataset.from_tensor_slices((raw_features, raw_labels))
        datatens = datatens.batch(batch_size).repeat(num_epochs)

        if shuffle:
            datatens = datatens.shuffle(10000)
        # Returns the next batch of data.
        feature_batch, label_batch = datatens.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return inputFunction

##Method name : createPredictInputFunction
##parama      : features    - The features to base predictions on.
#               labels      - The labels of the prediction examples.
#               batch_size  - Batch size
##return      : A function that returns features and labels for predictions.
def createPredictInputFunction(features, labels, batch_size):

    def inputFunction():
        raw_features = {"audioFeatures": features.values}
        raw_labels = np.array(labels)

        datatens = Dataset.from_tensor_slices((raw_features, raw_labels))
        datatens = datatens.batch(batch_size)

        # Returns the next batch of data.
        feature_batch, label_batch = datatens.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return inputFunction

##Method name : trainNNClassifier
##parama      : learning_rate - An `int`, the learning rate to use.
##                regularization_strength: A float, the regularization strength.
##              steps         - A non-zero `int`, the total number of training steps. A training step
##                              consists of a forward and backward pass using a single batch.
##              batch_size    - A non-zero `int`, the batch size.
##              hidden_units  - A `list` of int values, specifying the number of units in each layer.
##              training_examples - A `DataFrame` containing the training features.
##              training_labels - A `DataFrame` containing the training labels.
##              validation_examples - A `DataFrame` containing the validation features.
##              validation_labels - A `DataFrame` containing the validation labels.
##              model_Name - A `string` containing the model's name which is used when storing the loss curve and confusion       matrix plots.
##return      : A function that returns features and labels for predictions.
def trainNNClassifier(
        learning_rate,
        regularization_strength,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_labels,
        validation_examples,
        validation_labels,
        model_Name='no_Name'):
    periods = 10
    steps_per_period = steps / periods

    # Create the input functions.
    predict_traininginputFunction = createPredictInputFunction(
        training_examples, training_labels, batch_size)
    predict_validationinputFunction = createPredictInputFunction(
        validation_examples, validation_labels, batch_size)
    traininginputFunction = computeTrainInputFn(
        training_examples, training_labels, batch_size)

    # Create feature columns.
    feature_columns = computeFeatureColumns()

    # Create a DNNClassifier object.
    my_optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=learning_rate,
        l2_regularization_strength=regularization_strength  # can be swapped for l1 regularization
    )

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess loss metrics.
    print("Training model --->> ")
    print("LogLoss error computed on the validation data is :")
    training_errors = []
    validation_errors = []
    
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=traininginputFunction,
            steps=steps_per_period
        )

        # Use the current model to make predictions on both, the training and validation set.
        training_predictions = list(classifier.predict(input_fn=predict_traininginputFunction))
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validationinputFunction))
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Use predictions to compute training and validation errors.
        training_log_loss = metrics.log_loss(training_labels, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_labels, validation_pred_one_hot)

        # Print validation error of current model.
        print("  period %02d : %0.2f" % (period, validation_log_loss))

        # Store loss metrics so we can plot them later.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)

    print("Model training IS finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Compute predictions of final model.
    final_predictions = classifier.predict(input_fn=predict_validationinputFunction)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    # Evaluate predictions of final model.
    accuracy = metrics.accuracy_score(validation_labels, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    # plt.show()  # blocks execution
    plt.savefig('Results\\' + model_Name + '_loss_curve.png', bbox_inches='tight')
    plt.gcf().clear()

    # Create a confusion matrix.
    confusionMatrixPlot = metrics.confusion_matrix(validation_labels, final_predictions)

    # Normalize the confusion matrix by the number of samples in each class (rows).
    confusionMatrixPlot_normalized = confusionMatrixPlot.astype("float") / confusionMatrixPlot.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(confusionMatrixPlot_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig('Results\\' + model_Name + '_confusion_matrix.png', bbox_inches='tight')
    plt.gcf().clear()

    return classifier

    # for hyperparameter searching
def runAWSSession():
    # unpickle and prepare training data
    
    #code to read the data from s3 bucket
    session = boto3.session.Session(region_name='us-east-1')
    s3client = session.client('s3')

    response = s3client.get_object(Bucket='sound25', Key='Extracted_Features-notFold10_features.pkl')
    body_string = response['Body'].read()
    training_examples_data = cPickle.loads(body_string)
     
    training_examples = meanNormalizationOfData(training_examples_data)
    mean = np.mean(training_examples_data, axis=0)  
    std = np.std(training_examples_data, axis=0, ddof=1) 

    training_examples_data -= mean 
    training_examples_data /= std 
    training_examples = training_examples_data

    response = s3client.get_object(Bucket='sound25', Key='Extracted_Features-notFold10_labels.pkl')
    body_string = response['Body'].read()
    training_labels_data = cPickle.loads(body_string)
    training_labels = training_labels_data

    # unpickle and prepare validation data
    response = s3client.get_object(Bucket='sound25', Key='Extracted_Features-fold10_features.pkl')
    body_string = response['Body'].read()
    validation_examples_data = cPickle.loads(body_string)

    validation_examples = meanNormalizationOfData(validation_examples_data)
    mean = np.mean(validation_examples_data, axis=0)  
    std = np.std(validation_examples_data, axis=0, ddof=1) 

    validation_examples_data -= mean 
    validation_examples_data /= std 
    validation_examples = validation_examples_data
    
    response = s3client.get_object(Bucket='sound25', Key='Extracted_Features-fold10_labels.pkl')
    body_string = response['Body'].read()
    
    validation_labels_data = cPickle.loads(body_string)
    validation_labels = validation_labels_data

    for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        for regularization_strength in [0.0, 0.003, 0.03, 0.3]:
            print("##########################################################################")
            print("Learning rate:", learning_rate)
            print("Regularization:", regularization_strength)
            trainNNClassifier(
                learning_rate=0.003,
                regularization_strength=0.2,
                steps=10000,
                batch_size=32,
                hidden_units=[120],
                training_examples=training_examples,
                training_labels=training_labels,
                validation_examples=validation_examples,
                validation_labels=validation_labels)

def runSession():
    def runSession():
    # unpickle and prepare training data
        # unpickle and prepare training data
    training_examples_data = pd.read_pickle('Extracted_Features-notFold10_features.pkl')
    mean = np.mean(training_examples_data, axis=0)  
    std = np.std(training_examples_data, axis=0, ddof=1) 

    training_examples_data -= mean 
    training_examples_data /= std 
    training_examples = training_examples_data
    
    
    training_labels = pd.read_pickle('Extracted_Features-notFold10_labels.pkl')

    # unpickle and prepare validation data
    validation_examples_data = pd.read_pickle('Extracted_Features-fold10_features.pkl')
    mean = np.mean(validation_examples_data, axis=0)  
    std = np.std(validation_examples_data, axis=0, ddof=1) 

    validation_examples_data -= mean 
    validation_examples_data /= std 
    validation_examples = validation_examples_data
    
    
    validation_labels = pd.read_pickle('Extracted_Features-fold10_labels.pkl')
    
    for learning_rate in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        for regularization_strength in [0.0, 0.003, 0.03, 0.3]:
            print("##########################################################################")
            print("Learning rate:", learning_rate)
            print("Regularization:", regularization_strength)
            train_nn_classification_model(
                learning_rate=0.003,
                regularization_strength=0.2,
                steps=10000,
                batch_size=32,
                hidden_units=[120],
                training_examples=training_examples,
                training_labels=training_labels,
                validation_examples=validation_examples,
                validation_labels=validation_labels)
