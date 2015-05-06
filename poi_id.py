#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
from time import time
import pickle
import copy
import json
import wx
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,adjusted_rand_score
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn import cross_validation
from mytester import test_classifier, dump_classifier_and_data

# this function takes as input the enron dictionary and an email directory name in the format lastname-firstinitial possibly followed by digits.
# once it has the email address, it attempts to compare against the dictionary keys, which are in the format LASTNAME FIRSTNAME. 
# if a hit is found, the full key is returned so this keyword count attribute can be added to the dictionary
def find_person(dict,emaildir):
    cemail = emaildir.replace('-',' ')
    while (cemail[:-1].isdigit()):
        cemail = cemail[:-2]
    
    for d in dict:
        cd = d[:len(cemail)].lower()
        if cd == cemail:
            return d
    return None


# function to draw save to a png file a given chart
#
def Draw(pred, features, poi, mark_poi=True, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    fig = plt.figure()
    fig.suptitle(name, fontsize=14, fontweight='bold')

    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
#    plt.show()


# Please note placement of 'learning algorithms' does not reflect order of
# evaluation since this is implemented as a function and pre-defined before
# its use.
#
# function to run the currently selected feature set through our list of
# learning algorithms.  It returns the best_score metrics, the learned classifier
# 
def runlearning(data,new_features_list,data_dict):
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
    best_score = {"accuracy" : 0, "precision" : 0.3 , "recall" : 0.3 , "clf" : None}
    
    labels, features = targetFeatureSplit(data)

### split the feature set into training and testing sets, using cross validation, keeping a test size of 30% of the data
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.30, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler()
    features_train = min_max_scaler.fit_transform(features_train)
    features_test = min_max_scaler.fit_transform(features_test)
    
    
# start the training and testing
    
# Kmeans clustering
 
#===============================================================================
#     for cnum in [2]:
#         print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
#         clf = KMeans(n_clusters=cnum)
#         t0 = time()
#         clf.fit( features_train,labels_train )
#         print "training time:", round(time()-t0, 3), "s"
#         pred = clf.predict(features_test)
#         print "testing time:", round(time()-t0, 3), "s"
#         Draw(pred, features_test, labels_test, name="clusters"+str(cnum)+"_after_scaling.png", f1_name=new_features_list[1], f2_name=new_features_list[2])
# #        accuracy = accuracy_score(labels_test,pred)
#         prec = precision_score( labels_test, pred  , average='micro', pos_label=1)
#         recall = recall_score( labels_test, pred  , average='micro', pos_label=1)
#         print "precision score: ", prec 
#         print "recall score: ", recall 
# #        print "KMeans Train Clustering n="+str(cnum)+" accuracy: " + str(accuracy)
#         print "KMeans Classification Report: n="  + str(cnum) + ": "+ classification_report(labels_test,pred)
#         print "KMeans Confusion Matrix: n="  + str(cnum) + ": "
#         print confusion_matrix( labels_test,pred)
#         print "metrics.adjusted_rand_score:"
#         print adjusted_rand_score(labels_test, pred) 
#         (accuracy, prec, recall, f1, f2) = test_classifier(clf, data_dict, new_features_list)
#         if accuracy >= best_score["accuracy"] and prec >= best_score["precision"] and recall >= best_score["recall"]:
#             best_score["accuracy"] = accuracy
#             best_score["precision"] = prec
#             best_score["recall"] = recall
#             best_score["clf"] = clf
#===============================================================================
     

## Support Vector Machines, varying the Gamma parameter, and setting a 
# class_weight on our label value of 1.  Compared to observations, there
# are very few.
#    
    from sklearn import svm
    from sklearn.metrics import accuracy_score
     
#    GAMMAVALS = [0.06, 0.01, 0.006]
    GAMMAVALS = [0.01]
    for gammaval in GAMMAVALS:
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        print "gamma:" + str(gammaval)
         
#        clf = svm.SVC(kernel='rbf',C=1.0, gamma=gammaval,class_weight={1: 4})
        clf = svm.SVC(kernel='rbf',C=50.0, gamma=0.01, class_weight={1: 4})
#        clf = svm.SVC(kernel='rbf',C=10.0, gamma=1000, class_weight={1: 4}) 
        ### fit the classifier on the training features and labels
        t0 = time()
        clf.fit(features_train,labels_train)
        print "training time:", round(time()-t0, 3), "s"
         
        ### use the trained classifier to predict labels for the test features
        t0 = time()
        pred = clf.predict(features_test)
        print "testing time:", round(time()-t0, 3), "s"
        accuracy = accuracy_score(labels_test,pred)
        prec = precision_score( labels_test, pred , average='micro', pos_label=1)
        recall = recall_score( labels_test, pred  , average='micro', pos_label=1)
        print "precision score: ", prec
        print "recall score: ", recall
        print "SVM accuracy: " + str(accuracy)
        print "SVM Classification Report: "  + str(int(gammaval)) + ": "+ classification_report(labels_test,pred)
        print "SVM Confusion Matrix: "  + str(int(gammaval)) + ": "
        print confusion_matrix(labels_test,pred)
        (accuracy, prec, recall, f1, f2) = test_classifier(clf, data_dict, new_features_list)
        if accuracy >= best_score["accuracy"] and prec >= best_score["precision"] and recall >= best_score["recall"]:
            best_score["accuracy"] = accuracy
            best_score["precision"] = prec
            best_score["recall"] = recall
            best_score["clf"] = clf
    
    
#Naive Bayes Classifier
 
    #===========================================================================
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn.metrics import accuracy_score
    #   
    # print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    # ### create classifier
    # clf = GaussianNB()
    #   
    #   
    # ### fit the classifier on the training features and labels
    # t0 = time()
    # clf.fit(features_train,labels_train)
    # print "training time:", round(time()-t0, 3), "s"
    #   
    # ### use the trained classifier to predict labels for the test features
    # t0 = time()
    # pred = clf.predict(features_test)
    # print "testing time:", round(time()-t0, 3), "s"
    #   
    #   
    # ### calculate and return the accuracy on the test data
    # ### this is slightly different than the example, 
    # ### where we just print the accuracy
    # ### you might need to import an sklearn module
    # accuracy = accuracy_score(labels_test,pred)
    # prec = precision_score( labels_test, pred  , average='micro', pos_label=1)
    # recall = recall_score( labels_test, pred  , average='micro', pos_label=1)
    # print "precision score: ", prec
    # print "recall score: ", recall
    # print "Naive_Bayes: " + str(accuracy)
    # print "Naive_Bayes Classification Report: " + classification_report(labels_test,pred)
    # print "Naive_Bayes Confusion Matrix: "
    # print confusion_matrix(labels_test,pred)
    # (accuracy, prec, recall, f1, f2) = test_classifier(clf, data_dict, new_features_list)
    # if accuracy >= best_score["accuracy"] and prec >= best_score["precision"] and recall >= best_score["recall"]:
    #     best_score["accuracy"] = accuracy
    #     best_score["precision"] = prec
    #     best_score["recall"] = recall
    #     best_score["clf"] = clf
    #===========================================================================
    
# Decision Trees, varying the criterion between gini and entropy,
# and the min_samples_split

     
    #===========================================================================
    # from sklearn import tree
    #   
    # for cri in ['gini','entropy']:
    #     for cnum in [3,5]:
    #         print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    #         t0 = time()
    #         clf = tree.DecisionTreeClassifier(criterion=cri,min_samples_split=cnum)
    #         print "Classifier creation time:", round(time()-t0, 3), "s"
    #           
    #           
    #         ### fit the classifier on the training features and labels
    #         t0 = time()
    #         clf.fit(features_train,labels_train)
    #         print "training time:", round(time()-t0, 3), "s"
    #           
    #         ### use the trained classifier to predict labels for the test features
    #         t0 = time()
    #         pred = clf.predict(features_test)
    #         print "testing time:", round(time()-t0, 3), "s"
    #         ### calculate and return the accuracy on the test data
    #         ### this is slightly different than the example, 
    #         ### where we just print the accuracy
    #         ### you might need to import an sklearn module
    #         accuracy = accuracy_score(labels_test,pred)
    #         prec = precision_score( labels_test, pred  , average='micro', pos_label=1)
    #         recall = recall_score( labels_test, pred  , average='micro', pos_label=1)
    #         print "precision score: ", prec
    #         print "recall score: ", recall
    #         print "Decision Tree - " +cri + " " + str(cnum) + ": " + str(accuracy)
    #         print "Decision Tree Classification Report: "+cri + " "  + str(cnum) + ": " + classification_report(labels_test, pred)
    #         print "Decision Confusion Matrix: " +cri + " "  + str(cnum) + ": "
    #         print confusion_matrix(labels_test,pred)
    #         (accuracy, prec, recall, f1, f2) = test_classifier(clf, data_dict, new_features_list)
    #         if accuracy >= best_score["accuracy"] and prec >= best_score["precision"] and recall >= best_score["recall"]:
    #             best_score["accuracy"] = accuracy
    #             best_score["precision"] = prec
    #             best_score["recall"] = recall
    #             best_score["clf"] = clf
    #===========================================================================
    
    return best_score

#
# START OF MAINLINE CODE
#
### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# load in the user keyword count data that was generated from the
# scanning all the emails for specific keywords that may indicate
# knowledge of enron issues and/or mentions of important governmental
# people of influence.  This data is contained in the 
# your_authors_keyword_count.json file since it appears pkl files are not
# cross platform compatible.

### if you are creating any new features, you might want to do that here

for d in data_dict:
    if not data_dict[d].has_key("keyword_count"):
        data_dict[d]["keyword_count"] = 0
        try:
            if email_authors_keyword_count is None:
                json_data=open("your_authors_keyword_count.json").read()
                email_authors_keyword_count = json.loads(json_data)
        except:
            json_data=open("your_authors_keyword_count.json").read()
            email_authors_keyword_count = json.loads(json_data)

# and added to our data set

datachngd = False
try:
    if not email_authors_keyword_count is None:
        for email in email_authors_keyword_count:
            nm = find_person(data_dict,email)
            if data_dict.has_key(nm):
                data_dict[nm]["keyword_count"] = email_authors_keyword_count[email]
                datachngd = True
except:
    pass

# if we have changed the data set, save it back to pickle save file for
# subsequent runs
# disabled to ensure I don't change Udacity testing pkl file.
#===============================================================================
# if datachngd == True:
#     pickle.dump(data_dict, open("final_project_dataset.pkl", "w") )
#===============================================================================

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
target_label = "poi"
financial_features_list = [target_label, "salary","bonus","total_payments"]
email_features_list = [target_label, "from_this_person_to_poi","from_poi_to_this_person","keyword_count"]
#email_features_list = [target_label, "from_this_person_to_poi","from_poi_to_this_person"]
#email_features_list = [target_label, "salary","bonus","total_payments","keyword_count"]

### we suggest removing any outliers before proceeding further
### store to my_dataset for easy export below
del data_dict['TOTAL']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

my_dataset = data_dict
print data_dict.keys()


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
findata = featureFormat(my_dataset, financial_features_list,sort_keys=False)


emdata = featureFormat(my_dataset, email_features_list,sort_keys=False)

### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
flabels, ffeatures = targetFeatureSplit(findata)
elabels, efeatures = targetFeatureSplit(emdata)


## split into training and testing sets

### test_size is the percentage of events assigned to the test set (remainder go into training)
from sklearn import cross_validation
ffeatures_train, ffeatures_test, flabels_train, flabels_test = cross_validation.train_test_split(ffeatures, flabels, test_size=0.30, random_state=42)
efeatures_train, efeatures_test, elabels_train, elabels_test = cross_validation.train_test_split(efeatures, elabels, test_size=0.30, random_state=42)

# Scale the data here prior to PCA processing so PCA processing and machine learning deal with the
# same data.

scaler = preprocessing.StandardScaler().fit(ffeatures_train)
ffeatures_train = scaler.transform(ffeatures_train)
ffeatures_test = scaler.transform(ffeatures_test)

scaler = preprocessing.StandardScaler().fit(efeatures_train)
efeatures_train = scaler.transform(efeatures_train)
efeatures_test = scaler.transform(efeatures_test)

## use PCA on all available financial data features and let it decide
## what components are important.

t0 = time()
#pca = RandomizedPCA(n_components=2, whiten=True).fit(features_train)
pca = PCA(n_components=2, whiten=True).fit(ffeatures_train)
print "done in %0.3fs" % (time() - t0)
print "explained_variance_ratio_:" + str(pca.explained_variance_ratio_)
 
new_ffeatures_list = [target_label]
if (pca.explained_variance_ratio_[0] < 0.75):
    new_ffeatures_list[1:] = financial_features_list[1:]
else:
    print pca.components_[0]
    first_pc = pca.components_[0]
    print first_pc
    coeffound = False
    for i in xrange(len(first_pc)):
        if (first_pc[i] > 0.0000000001):
            coeffound = True
            break
    if coeffound == False:
        first_pc = pca.components_[0] * -1
        
    first_pc = sorted(first_pc,reverse=True)
    for i in xrange(len(first_pc)):
        if (first_pc[i] > 0.0000000001):
            new_ffeatures_list.append(financial_features_list[i+1])
            print "1st PC: " + financial_features_list[i+1] + " has PCA coef: " + str(first_pc[i])
     

print "Features Chosen:"
print new_ffeatures_list

## use PCA on all available email data features and let it decide
## what components are important.

pca = PCA(n_components=2, whiten=True).fit(efeatures_train)
print "done in %0.3fs" % (time() - t0)
print "explained_variance_ratio_:" + str(pca.explained_variance_ratio_)
new_efeatures_list = [target_label]
if (pca.explained_variance_ratio_[0] < 0.75):
    new_efeatures_list[1:] = email_features_list[1:]
else:
    print pca.components_[0]
    first_pc = pca.components_[0]
    print first_pc
    coeffound = False
    for i in xrange(len(first_pc)):
        if (first_pc[i] > 0.0000000001):
            coeffound = True
            break
    if coeffound == False:
        first_pc = pca.components_[0] * -1
        
    first_pc = sorted(first_pc,reverse=True)
    for i in xrange(len(first_pc)):
        if (first_pc[i] > 0.0000000001):
            new_efeatures_list.append(email_features_list[i+1])
            print "1st PC: " + email_features_list[i+1] + " has PCA coef: " + str(first_pc[i])
         
 
print "Features Chosen:"
print new_efeatures_list


# we have the updated feature lists in case PCA analysis dropped any

findata = featureFormat(my_dataset, new_ffeatures_list,sort_keys=False )
emdata = featureFormat(my_dataset, new_efeatures_list,sort_keys=False )

### machine learning goes here!
### please name your classifier clf for easy export below

#fbest_score = runlearning(findata,new_ffeatures_list,my_dataset)
fbest_score = None
ebest_score = runlearning(emdata,new_efeatures_list,my_dataset)

if fbest_score != None and fbest_score["accuracy"] > ebest_score["accuracy"] and fbest_score["precision"] > ebest_score["precision"] and fbest_score["recall"] > ebest_score["recall"]:
    clf = fbest_score["clf"]
    new_features_list = new_ffeatures_list
else:
    clf = ebest_score["clf"]
    new_features_list = new_efeatures_list
    
# dump pkl files for tester.py
#
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(new_features_list, open("my_feature_list.pkl", "w") )

print "# of observations: ", len(data_dict)
print "features used: ", new_features_list
test_classifier(clf, data_dict, new_features_list)

