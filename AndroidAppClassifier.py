"""
    ----------------------------------------------------------------------------------------
    AndroidAppClassifier (training_file, testing_file, split_data=False, group_game=False):
    ----------------------------------------------------------------------------------------
    This classifier, classifies Android apps on Google Play to appropriate categories,
    using Naive Bayes Classifier (MultinomialNB and BernoulliNB)

    Parameters
    ----------
    training_file : string
        The training file for the classifier

    testing_file : string
        The testing file for the classifier

    split_data : boolean
        If True, the classifier uses the training file alone and splits the file into two
        (n,n), using one as the training set and the other as the testing set. The split
        value is adjusted manually

        If false, the classifier uses the training file and test file specified

    group_game : boolean
        If True, Game categories ('GAME_BOARD' 'GAME_CARD' 'GAME_CASINO' 'GAME_CASUAL'
        'GAME_EDUCATIONAL' 'GAME_MUSIC' 'GAME_PUZZLE' 'GAME_RACING' 'GAME_ROLE_PLAYING'
        'GAME_SIMULATION' 'GAME_SPORTS' 'GAME_STRATEGY' 'GAME_TRIVIA' 'GAME_WORD') are
        all grouped as (GAMES). This improves the performance of the classifier because
        Games have very similar features, so the classifier might misclassify some games.

        If False, the classifier uses all Game categories in the data set.

"""
from __future__ import print_function
import logging
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.learning_curve import learning_curve
import uuid
import re
import os
import sys
import multiprocessing
from pandas_confusion import ConfusionMatrix
import pandas as pd
import warnings

class Tee(object):
    # Help print console text to output file
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

class AndroidAppClassifier:
    def __init__(self, training_file, testing_file, split_data=False, group_game=False):
        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        self.training_file = training_file  # Get the training file
        self.test_file = testing_file  # Get the test file

                # create a folder for our result if it doesnt exist
        if not os.path.exists('results'):
            os.makedirs('results')

        # redirect output to file
        f = open('results/output.txt', 'w')
        original = sys.stdout
        sys.stdout = Tee(sys.stdout, f)

        # adjust the console to show on all in one row
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        # ignore numpy 1.8.0 deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        #
        #   all vectors to store our training and test categories and data
        #   self.X_train, self.X_test = (training data, test data)
        #   self.y_train, self.y_test = (training classes, test classes)
        #   self.feature_names, self.categories = (features for the training data, categories found in training data)
        #   self.classifierResults = results obtained from the classifier
        #
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names, self.categories, self.classifierResults = [], [], [], [], [], [], []
        self.split_data = split_data
        self.group_game = group_game

    def wordcount(self, value):
        # Find all non-whitespace patterns.
        list = re.findall("(\S+)", value)
        # Return length of resulting list.
        return len(list)

    def normalizeData(self, datafile):
        """
        Cleans up the input file and regroup features

        Parameters
        ----------
        datafile : The input file for normalization "string"
            datafile is a CSV file with the following format:
                Category,AppName,ContentRating,IsFree,HaveInAppPurchases,Description
        returns:
            Category,data
        """

        # Google play accepts maximum of 4000 characters ~ 2000 words.
        # An app with 100 words is descriptive enough

        MIN_DESC = 100                     # minimum number of words in a description

        skip_lowDesc = False              # skip apps with low descriptions
        skip_lowCategory = False              # skip categories with low number of apps
        classify_only_games  = False         # classify only games
        classify_only_others  = False         # other apps apart from games
        # categoroes to skip
        skip_cats = ['COMICS', 'LIBRARIES_AND_DEMO', 'GAME_MUSIC', 'GAME_WORD']

        out = [] #  data output vector
        IsFree, HaveInAppPurchases = '', '' #   detect inapp purchase and free in file
        data_result = np.genfromtxt(open(datafile, 'r'), delimiter=',', dtype=None, autostrip=True)

        data_result = data_result[1:, ] #   skip the headers

        #   if InAppPurchase or IsFree exist, combine it with the description
        for row in data_result:
            # skip apps with few description (words)
            if skip_lowDesc:
                if self.wordcount(row[5]) < MIN_DESC:
                    continue
            # skip low categories
            if skip_lowCategory:
                if row[0] in skip_cats:
                    continue
            # classify other apps apart from games
            if classify_only_others:
                if row[0].find('GAME_') != -1:
                    continue
            # classify only games
            if classify_only_games:
                if row[0].find('GAME_') == -1:
                    continue

            if (row[3] == 'True'):
                IsFree = ' IsFree '
            if (row[4] == 'True'):
                HaveInAppPurchases = ' HaveInAppPurchases '

            new_class = row[0] # the category
            #   look for Games and re-categorize all games as one if group_game is enabled
            if self.group_game:
                if row[0].find('GAME_') != -1:
                    new_class = 'GAMES'


            new_desc = row[5] + IsFree + HaveInAppPurchases + row[1] + row[2] # the new description, combining all

            out.append([new_class, new_desc]) # append the result to out.

        return np.array(out)

    def buildClassifier(self):
        """
        Builds up training and test set data and performs TiDf vectorizer
        using bag-of-words method to extract features

        """

        print('Loading data from file....')
        data_train = self.normalizeData(self.training_file) # load the normalized training data
        data_test = self.normalizeData(self.test_file) # load the normalized testing data

        print('Data Loaded Successfully')
        print("Loading Android Apps categories:")

        #   If slip_data is enabled, use only the training set and split it to get the test set
        if self.split_data:
            #   test_size : float, int, or None (default is None)
            #   split the data using the (test_size). If test_size is not specified, use 25% of data for testing
            X_train, X_test, y_train, y_test = train_test_split(data_train[:, 1:].ravel(), data_train[:, 0],
                                                                test_size=.2, random_state=0)
            self.categories = np.unique(y_train)                # get unique categories from classes in training file
            print("Categories: ", self.categories)
            print("%d Apps (Training set)" % len(X_train))      # get the number of training set
            print("%d Apps (Test set)" % len(X_test))           # get the number of test set
            print("%d Categories" % len(self.categories))       # count categories
            print()
        else:
            #   use the training and test files specified for analysis
            self.categories = np.unique(data_train[:, 0])       # get unique categories from classes in training file
            print("Categories: ", self.categories)
            print("%d Apps (Training set)" % len(data_train))   # get the number of training set
            print("%d Apps (Test set)" % len(data_test))        # get the number of test set
            print("%d Categories" % len(self.categories))       # count categories
            print()

        if self.split_data:
            # use the splitted training file if split_data is true
            self.y_train, self.y_test = y_train, y_test
        else:
            # split a training set and a test set
            self.y_train, self.y_test = data_train[:, 0], data_test[:, 0]

        #============================================
        #    For Training Set
        #============================================
        print("Extracting features from the TRAINING data using a sparse vectorizer")
        t0 = time()

        # remove words occuring in only two document or at least 70% of the documents are removed
        vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.7, min_df=.0005,   # max_df=0.70, min_df=3,   #max_features=1000, # extract features using bag of words
                                     stop_words='english')             # stop_words='english' removes common stop words

        if self.split_data:
            self.X_train = vectorizer.fit_transform(X_train)           # fit data using the splitted training set
        else:
            self.X_train = vectorizer.fit_transform(data_train[:, 1:].ravel()) # fit data using the training file

        duration = time() - t0                                          # calculate the duration
        print("done in %fs " % (duration))
        n_features = self.X_train.shape[1]
        self.n_feature10 = n_features *.02                              # select 2% of features
        print("n_samples: %d, n_features: %d" % self.X_train.shape)     # get the number of features and samples from training set
        print()

        #============================================
        #    For Test Set
        #============================================
        print("Extracting features from the TEST data using sparse vectorizer")
        t0 = time()
        if self.split_data:
            self.X_test = vectorizer.transform(X_test)                      # fit data using the splitted test set
        else:
            self.X_test = vectorizer.transform(data_test[:, 1:].ravel())    # fit data using the test file

        duration = time() - t0
        print("done in %fs " % (duration))
        print("n_samples: %d, n_features: %d" % self.X_test.shape)      # get the number of features and samples from test set
        print()

        # mapping from integer feature name to original token string
        self.feature_names = vectorizer.get_feature_names()
        # print ("Features: ", feature_names)
        #print("Normalized Training Set:")
        #print(self.X_train.toarray())       # tidf transformed data of all features in training set

        if self.feature_names:
            self.feature_names = np.asarray(self.feature_names) # store all feature names


    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and traning learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : integer, cross-validation generator, optional
            If an integer is passed, it is the number of folds (defaults to 3).
            Specific cross-validation objects can be passed, see
            sklearn.cross_validation module for the list of possible objects

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt


    def benchmark(self, clf):
        """
        Prints out results of all classifier used

        Parameters
        ----------
        clf : The classifier to benchmark (MultinonialNB and Ber...)
        returns:
           clf_descr, score, train_time, test_time
                the classifier description, score, training time and testing time to plot
        """
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(self.X_train, self.y_train)         # fit the classifier with the features/ train the classifier
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)    # get the duration

        t0 = time()
        pred = clf.predict(self.X_test)             # perform prediction
        print("Predictions: ", pred)
        test_time = time() - t0
        print("Test time:  %0.3fs" % test_time)     # show estimated time for prediction

        score = metrics.accuracy_score(self.y_test, pred)   # calculate the accuracy on the test file
        print("Accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])                # prints the dimentionality of the data
            print("density: %f" % density(clf.coef_))

            print("top 10 keywords per category:")                          # gets the Top10 features per category
            for i, category in enumerate(self.categories):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print("%s: %s"
                      % (category, " ".join(self.feature_names[top10])))
            print()

        # prints the classification report for a regular test
        print("Classification report:")
        print(metrics.classification_report(self.y_test, pred,
                                            target_names=self.categories))

        # prints the confusion matrix for a regular test
        #print("Confusion matrix:")
        #print(metrics.confusion_matrix(self.y_test, pred))
        #y_actu = pd.Series(self.y_test, name='Actual')
        #y_pred = pd.Series(pred, name='Predicted')
        #df_confusion = pd.crosstab(y_actu, y_pred)
        #print (df_confusion)
        cm = ConfusionMatrix(self.y_test, pred)
        cm.print_stats()


        print()
        clf_descr = str(clf).split('(')[0]

        # if split_data is enabled, perform all forms of cross validation
        if self.split_data:

            processes = [self.kfoldCV, self.shuffleCV, self.recurCV] # our cross validations
            #processes = [self.kfoldCV]
            for p in processes:

                # use multiprocessing to make computation faster
                # ============= K-Fold Validation =============
                # ============ Shuffle Split cross validation (learning Curve) ================
                # ============ Recursive feature elimination ================
                self.process = multiprocessing.Process(target=p, args=(clf,))
                self.process.start()

        return clf_descr, score, train_time, test_time # return for regular splitting between test and training file

    def shuffleCV(self,clf):
        #print ('Shuffle Process Unique Id: {0}'.format(uuid.uuid1()))
        # ============ Shuffle Split cross validation (learning Curve) ================
        t0 = time()
        title = "Learning Curves (Naive Bayes) " + str(clf).split('(')[0] # prints the name of classifier also
        # Cross validation with 20 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(len(self.y_train), n_iter=20,
                                           test_size=0.2, random_state=0)

        # plots a graph showing the learning curve for the test and training data split the job to 4 threads
        plt = self.plot_learning_curve(clf, title, self.X_train, self.y_train, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
        plt.draw()
        plt.savefig("results/"+str(clf).split('(')[0] + '_shuffleCVlearningCurve')
        ss_time = time() - t0
        print("(", str(clf).split('(')[0], ") Shuffle Split time:  %0.3fs" % ss_time)      # orints estimated time

        # ============ Shuffle Split cross validation ================

    def kfoldCV(self,clf):
        #print ('Kfold Process Unique Id: {0}'.format(uuid.uuid1()))
        # ============= K-Fold Validation =============
        t0 = time()

        # perform k-fold cross validation
        # n_folds=10, shuffle=False, random_state=None (use random shuffle for 2-fold cross validation)
        kfold = cross_validation.KFold(n=len(self.y_train), n_folds=10, shuffle=False,
                                     random_state=None)
        CVscore = cross_validation.cross_val_score(clf, self.X_train, self.y_train, cv=kfold)
        print("(", str(clf).split('(')[0], ") K-Fold Scores:  ", CVscore)          # print accuracy for all n-folds
        cv_time = time() - t0
        print("(", str(clf).split('(')[0], ") K-Fold time:  %0.3fs" % cv_time)     # print estimated time
        # ============= K-Fold Validation =============

    def recurCV(self,clf):
        #print ('RFECV Unique Id: {0}'.format(uuid.uuid1()))
        # ============ Recursive feature elimination using StratifiedKFold ================
        # we use this to estimate the best number of features for the data.
        # this looks for the highest score for n number of features and plots a graph showing
        # the accuracy for n features.
        t0 = time()
        # Create the RFE object and compute a cross-validated score. Remove 10% features in each iteration of Strat cross validation
        # Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole.
        # For example in a binary classification problem where each class comprises 50% of the data, it is best to arrange the
        # data such that in every fold, each class comprises around half the instances.

        # This will iterate 50 times in the dataset
        rfecv = RFECV(estimator=clf, step=self.n_feature10, cv=cross_validation.StratifiedKFold(self.y_train, n_folds=2),
                      scoring='accuracy')
        rfecv.fit(self.X_train, self.y_train)

        print("(", str(clf).split('(')[0], ") RFE Optimal number of features : %d" % rfecv.n_features_)    # display the perfect number of features for optimum accuracy

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.title("Recursive Feature Elimination for " + str(clf).split('(')[0])
        plt.xlabel("Number of features selected in %")
        plt.ylabel("StratifiedKFold Cross validation score")
        plt.plot(range(1, len(rfecv.grid_scores_)*2, 2 ), rfecv.grid_scores_)
        plt.draw()
        plt.savefig("results/"+str(clf).split('(')[0] + '_recurCV')
        rf_time = time() - t0
        print("(", str(clf).split('(')[0], ") RFE time:  %0.3fs" % rf_time)        # prints the time consumed for the analysis

        # ============ Recursive feature elimination ================

    def classifyNB(self):
        """
        Train sparse Naive Bayes classifiers

        """
        print('=' * 80)
        print("Naive Bayes")

        self.classifierResults.append(self.benchmark(MultinomialNB(alpha=.01)))    # MultinomialNB classifier
        self.classifierResults.append(self.benchmark(BernoulliNB(alpha=.01)))      # BernoulliNB classifier

    def plotData(self):
        """
        Plot data for the two classifiers used

        """
        indices = np.arange(len(self.classifierResults))    # the x locations for the groups
        width = 0.27  # the width of the bars
        results = [[x[i] for x in self.classifierResults] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / 60 # in minutes
        test_time = np.array(test_time) / 60 # in minutes

        fig = plt.figure()
        ax = fig.add_subplot(111)

        rects1 = ax.bar(indices, score, width, label="score", color='r')                          # bar for score
        rects2 = ax.bar(indices + width, training_time, width, color='g')     # bar for training time
        rects3 = ax.bar(indices + width*2, test_time, width, color='y')             # bar for testing time

        ax.set_ylabel('Score / Time(minutes)')  # label our y axis
        ax.set_xticks(indices + width*2 - 0.13)
        ax.set_xticklabels(('MultinomialNB', 'BernoulliNB'))  # Our x axis label
        ax.legend((rects1[0], rects2[0], rects3[0]), ('Accuracy', 'Training Time', 'Testing Time'), loc=1)  # draw the legend

        # this method auto labels our graph appropriately
        def autolabel(rects):
            totalHt = []
            for rect in rects:
                h = rect.get_height()
                totalHt.append(h)
                ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * h, '%.3f' % h,
                        # where we would place the legend
                        ha='center', va='bottom')
            return totalHt

        maxY = []
        maxY.append(autolabel(rects1))
        maxY.append(autolabel(rects2))
        maxY.append(autolabel(rects3))

        plt.ylim(0, 1)  # increase y axis a little to prevent overlapping

        plt.title("Classifier Accuracy")
        plt.draw()

        plt.savefig('results/classifier_results')


if __name__ == "__main__":

    training_file = 'data/PlayStoreDataTopDevelopers.csv'                           # Training file path
    testing_file = 'data/PlayStoreDataTopDevelopers.csv'                            # Testing file path

    """
    AndroidAppClassifier (training_file, testing_file, split_data=False, group_game=False)

    Parameters
    ----------
    training_file : string
        The training file for the classifier

    testing_file : string
        The testing file for the classifier

    split_data : boolean
        If True, the classifier uses the training file alone and splits the file into two
        (n,n), using one as the training set and the other as the testing set. The split
        value is adjusted manually

        If false, the classifier uses the training file and test file specified

    group_game : boolean
        If True, Game categories ('GAME_BOARD' 'GAME_CARD' 'GAME_CASINO' 'GAME_CASUAL'
        'GAME_EDUCATIONAL' 'GAME_MUSIC' 'GAME_PUZZLE' 'GAME_RACING' 'GAME_ROLE_PLAYING'
        'GAME_SIMULATION' 'GAME_SPORTS' 'GAME_STRATEGY' 'GAME_TRIVIA' 'GAME_WORD') are
        all grouped as (GAMES). This improves the performance of the classifier because
        Games have very similar features, so the classifier might mis-classify some games.

        If False, the classifier uses all Game categories in the data set.
    """
    BuildModel = AndroidAppClassifier(training_file, testing_file, True, False)     # Call the class
    BuildModel.buildClassifier()                                                    # Train
    BuildModel.classifyNB()                                                         # Classify
    if BuildModel.split_data:
        BuildModel.process.join()                                                       # wait for all process to finish before plotting
    BuildModel.plotData()                                                           # Plot results
