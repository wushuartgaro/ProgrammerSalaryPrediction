# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 04:08:13 2017

@author: Hard-
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import csv
from sklearn import cross_validation
from operator import itemgetter
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 11,9

#set color    
W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red
G  = '\033[32m' # green
O  = '\033[33m' # orange
B  = '\033[34m' # blue
P  = '\033[35m' # purple
#For converting String Data of all the columns in classes [0,1]
#update data_clean with transformed values of all the columns
def Preprocessing(data_clean):
    le = preprocessing.LabelEncoder()

    le.fit(data_clean.ProgramHobby)
    data_clean.loc[:,'ProgramHobby'] = le.transform(data_clean.ProgramHobby)
    le.fit(data_clean.Country)
    data_clean.loc[:,'Country'] = le.transform(data_clean.Country)
    le.fit(data_clean.University)
    data_clean.loc[:,'University'] = le.transform(data_clean.University)
    le.fit(data_clean.EmploymentStatus)
    data_clean.loc[:,'EmploymentStatus'] = le.transform(data_clean.EmploymentStatus)
    le.fit(data_clean.FormalEducation)
    data_clean.loc[:,'FormalEducation'] = le.transform(data_clean.FormalEducation)
    le.fit(data_clean.MajorUndergrad)
    data_clean.loc[:,'MajorUndergrad'] = le.transform(data_clean.MajorUndergrad)
    le.fit(data_clean.HomeRemote) 
    data_clean.loc[:,'HomeRemote'] = le.transform(data_clean.HomeRemote)
    le.fit(data_clean.CompanySize)
    data_clean.loc[:,'CompanySize'] = le.transform(data_clean.CompanySize)
    le.fit(data_clean.CompanyType)
    data_clean.loc[:,'CompanyType'] = le.transform(data_clean.CompanyType)
    le.fit(data_clean.HaveWorkedLanguage)
    data_clean.loc[:,'HaveWorkedLanguage'] = le.transform(data_clean.HaveWorkedLanguage)
    le.fit(data_clean.YearsProgram)
    data_clean.loc[:,'YearsProgram'] = le.transform(data_clean.YearsProgram)
    le.fit(data_clean.YearsCodedJob)
    data_clean.loc[:,'YearsCodedJob'] = le.transform(data_clean.YearsCodedJob)
    le.fit(data_clean.DeveloperType)
    data_clean.loc[:,'DeveloperType'] = le.transform(data_clean.DeveloperType)
    le.fit(data_clean.PronounceGIF)
    data_clean.loc[:,'PronounceGIF'] = le.transform(data_clean.PronounceGIF)
    le.fit(data_clean.TabsSpaces)
    data_clean.loc[:,'TabsSpaces'] = le.transform(data_clean.TabsSpaces)
    le.fit(data_clean.Gender)
    data_clean.loc[:,'Gender'] = le.transform(data_clean.Gender)
    le.fit(data_clean.HighestEducationParents)
    data_clean.loc[:,'HighestEducationParents'] = le.transform(data_clean.HighestEducationParents)
    le.fit(data_clean.Race)
    data_clean.loc[:,'Race'] = le.transform(data_clean.Race)
    
    #select predictor variables and target variable as separate data sets
    predvar= data_clean[['ProgramHobby','Country','University','EmploymentStatus','FormalEducation',
                     'MajorUndergrad','HomeRemote','CompanySize','CompanyType','HaveWorkedLanguage','YearsProgram',
                     'YearsCodedJob','DeveloperType','PronounceGIF','TabsSpaces','Gender','HighestEducationParents','Race']]

    predictors=predvar.copy()
                    
    predictors['ProgramHobby']=preprocessing.scale(predictors['ProgramHobby'].astype('float64'))
    predictors['Country']=preprocessing.scale(predictors['Country'].astype('float64'))
    predictors['University']=preprocessing.scale(predictors['University'].astype('float64'))
    predictors['EmploymentStatus']=preprocessing.scale(predictors['EmploymentStatus'].astype('float64'))
    predictors['FormalEducation']=preprocessing.scale(predictors['FormalEducation'].astype('float64'))
    predictors['MajorUndergrad']=preprocessing.scale(predictors['MajorUndergrad'].astype('float64'))
    predictors['HomeRemote']=preprocessing.scale(predictors['HomeRemote'].astype('float64'))
    predictors['CompanySize']=preprocessing.scale(predictors['CompanySize'].astype('float64'))
    predictors['CompanyType']=preprocessing.scale(predictors['CompanyType'].astype('float64'))
    predictors['HaveWorkedLanguage']=preprocessing.scale(predictors['HaveWorkedLanguage'].astype('float64'))
    predictors['YearsProgram']=preprocessing.scale(predictors['YearsProgram'].astype('float64'))
    predictors['YearsCodedJob']=preprocessing.scale(predictors['YearsCodedJob'].astype('float64'))
    predictors['DeveloperType']=preprocessing.scale(predictors['DeveloperType'].astype('float64'))
    predictors['PronounceGIF']=preprocessing.scale(predictors['PronounceGIF'].astype('float64'))
    predictors['TabsSpaces']=preprocessing.scale(predictors['TabsSpaces'].astype('float64'))
    predictors['Gender']=preprocessing.scale(predictors['Gender'].astype('float64'))
    predictors['HighestEducationParents']=preprocessing.scale(predictors['HighestEducationParents'].astype('float64'))
    predictors['Race']=preprocessing.scale(predictors['Race'].astype('float64'))
 
    return predictors

#Lasso Regression
def LassoPrediction(X_train, X_test, Y_train):
    lasso = Lasso(alpha=0.1, normalize=True, max_iter=1e5)
    lasso.fit(X_train, Y_train)
    return lasso

#Random Forest
def RandomForestPrediction(X_train, X_test, Y_train):
    y, _ = pd.factorize(Y_train)
    forest = RFC(max_leaf_nodes=40,min_samples_leaf=9,n_estimators=7,min_weight_fraction_leaf=0.1,min_samples_split=8,max_depth=9)    
    """
    forest = RFC(n_jobs=1)
    forest = RFC(n_jobs=1,n_estimators=200,max_features="sqrt",random_state=50,min_samples_leaf=5,min_samples_split=100,max_depth=9)
    """    
    forest.fit(X_train, y)
    return forest

#Gradient Boosting
def GradientBoosting(X_train, X_test, Y_train):
    y, _ = pd.factorize(Y_train)
    gbm0 = GradientBoostingClassifier(learning_rate=0.01)    
    """
    gbm0 = GradientBoostingClassifier(random_state=10,learning_rate=0.01,n_estimators=200,min_samples_split=100,min_samples_leaf=5,max_depth=9,max_features="sqrt")
    """    
    gbm0.fit(X_train, y)
    return gbm0

#plot confusion matrix, display accuracy,precision,recall,f1score
def plotConfusionMatrix(model,Y_test):
    print "Confusion Matrix:"
    for index,p in enumerate(model):
        if p==0:
            model[index] = 20000
        elif p==1:
            model[index] = 40000
        elif p==2:
            model[index] = 60000
        elif p==3:
            model[index] = 80000
        else:
            model[index] = 100000
            
    print pd.crosstab(index=Y_test, columns=model, rownames=['actual'], colnames=['preds'])
    print '\nClasification report:\n', classification_report(Y_test, model)
    print "Overall Accuracy: ",round(accuracy_score(Y_test, model),2)    
    """
    print "Overall Precision: ",round(precision_score(Y_test, model, average="macro"),2)
    print "Overall Recall: ",round(recall_score(Y_test, model, average="macro"),2) 
    print "Overall F1 Score: ",round(f1_score(Y_test, model, average="macro"),2)
    """
    
def feature_importances(X_train,Y_train,data_clean,forestModel):
    names=['ProgramHobby','Country','University','EmploymentStatus','FormalEducation','MajorUndergrad','HomeRemote','CompanySize','CompanyType','HaveWorkedLanguage','YearsProgram','YearsCodedJob','DeveloperType','PronounceGIF','TabsSpaces','Gender','HighestEducationParents','Race']
    df = pd.DataFrame(data_clean, columns=names)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .80
    df['Salary'] = data_clean.Salary
    
    features = df.columns[0:19]
    importances = forestModel.feature_importances_
    indices = np.argsort(importances)

    plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), features[indices])
    plt.xlabel('Relative Importance')
    plt.show()

def trainAndSaveModels():
    """
    #Load the dataset
    data = pd.read_csv("survey_results_public_.csv")
    print "\nPreprocessing the data..."
    #Data Management
    data_clean = data.dropna()
    predictors = Preprocessing(data_clean)
    print "\nSaving data..."
    #save preprocessed data
    pickle.dump(data_clean,open('data_clean.sav','wb'))
    pickle.dump(predictors,open('predictors.sav','wb'))
    #set salary as target
    target = data_clean.Salary
    print "\nSplitting data into train and test samples..."
    # split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target,test_size=.2)
    #save train and test data    
    pickle.dump(X_train,open('X_train.sav','wb'))
    pickle.dump(X_test,open('X_test.sav','wb'))
    pickle.dump(Y_train,open('Y_train.sav','wb'))
    pickle.dump(Y_test,open('Y_test.sav','wb'))
    """
    print "\nTraining models..."

    #Use the best-performed train and test splitted data 
    X_train = pickle.load(open('X_train.sav','rb'))
    X_test = pickle.load(open('X_test.sav','rb'))
    Y_train = pickle.load(open('Y_train.sav','rb'))
    
    #train models
    lassoModel = LassoPrediction(X_train, X_test, Y_train)
    forestModel = RandomForestPrediction(X_train, X_test, Y_train)
    boostingModel = GradientBoosting(X_train, X_test, Y_train)
    
    #save the modes
    pickle.dump(lassoModel,open('lasso_Model.sav','wb'))
    pickle.dump(forestModel,open('forest_Model.sav','wb'))
    pickle.dump(boostingModel,open('sgb_Model.sav','wb'))

def initialization(choice):
    
    #Read the preprocessed data
    data_clean = pickle.load(open('data_clean.sav','rb'))
    predictors = pickle.load(open('predictors.sav','rb'))
    
    #Read the pre-defined train test data
    X_train = pickle.load(open('X_train.sav','rb'))
    X_test = pickle.load(open('X_test.sav','rb'))
    Y_train = pickle.load(open('Y_train.sav','rb'))
    Y_test = pickle.load(open('Y_test.sav','rb'))

    #Read the pre-trained models                                                              s   
    lassoModel = pickle.load(open('lasso_Model.sav','rb')) 
    forestModel = pickle.load(open('forest_Model.sav','rb'))
    boostingModel = pickle.load(open('sgb_Model.sav','rb')) 
    
    #predict the result    
    lassoResult = lassoModel.predict(X_test)
    forestResult = forestModel.predict(X_test)    
    boostingResult = boostingModel.predict(X_test)
    
    #categorize the real value predicted by lasso regression model
    for index,p in enumerate(lassoResult):
        if 0<=p<=20000:
            lassoResult[index] = 0
        elif p<=40000:
            lassoResult[index] = 1
        elif p<=60000:
            lassoResult[index] = 2
        elif p<=80000:
            lassoResult[index] = 3
        else:
            lassoResult[index] = 4

    
    if(choice == 2):
        #display the performance of each model
        print(B+"\n================\nLasso Regression\n================\n"+W)
        plotConfusionMatrix(lassoResult,Y_test)
        print(B+"\n=============\nRandom Forest\n=============\n"+W)
        plotConfusionMatrix(forestResult,Y_test)
        print(B+"\n============================\nGradient Boosting\n============================\n"+W)
        plotConfusionMatrix(boostingResult,Y_test)
        back = raw_input("\nEnter 'Y' for going back to main menu or else for exit: ")
        if(back=='Y'or back=='y'):
           main()
        else:
            print(B+"\nBye Bye!"+W)
    
    elif(choice == 3):
        def lasso_regression(data,alpha, models_to_plot={}):
            #Fit the model
            lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
            lassoreg.fit(data,Y_train)
            y_pred = lassoreg.predict(data)
    
            #Check if a plot is to be made for the entered alpha
            if alpha in models_to_plot:
                plt.subplot(models_to_plot[alpha])
                plt.ylim(10000,120000)
                plt.tight_layout()
                plt.plot(data,y_pred)
                plt.plot(data,Y_train,'.')
                plt.title('Plot for alpha: %.3g'%alpha)
                
                #Return the result in pre-defined format
                rss = sum((y_pred-Y_train)**2)
                ret = [rss]
                ret.extend([lassoModel.intercept_])
                ret.extend(lassoModel.coef_)
                return ret
                
        #Define the alpha values to test
        alpha_lasso = [1e-5,0.1,1,20,50,150]
        
        #Initialize the dataframe to store coefficients
        col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,19)]
        ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,6)]
        coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)
    
        #Define the models to plot
        models_to_plot = {1e-5:231,0.1:232,1:233,20:234,50:235,150:236}
        #Iterate over the 10 alpha values:
        for i in range(len(alpha_lasso)):
            print "Plot ",i+1," Prepared..."
            coef_matrix_lasso.iloc[i,] = lasso_regression(X_train,alpha_lasso[i], models_to_plot)
        plt.show()
        
        # specify the lasso regression model
        model=LassoLarsCV(cv=10, precompute=False).fit(X_train,Y_train)

        # print variable names and regression coefficients
        print "\nVariable Names and their respective regression coefficient\n"
        print dict(zip(predictors.columns, model.coef_))
        
        #plot mean square error for each fold
        m_log_alphascv = -np.log10(model.cv_alphas_)
        plt.figure()
        plt.plot(m_log_alphascv, model.cv_mse_path_, unicode(':', 'utf-8'))
        plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',label=unicode('Average across the folds', 'utf-8'), linewidth=2)
        plt.axvline(-np.log10(model.alpha_), linestyle='-', color='k',label=unicode('alpha CV', 'utf-8'))
        plt.legend()
        plt.xlabel(unicode('-log(alpha)', 'utf-8'))
        plt.ylabel(unicode('Mean squared error', 'utf-8'))
        plt.title(unicode('Mean squared error on each fold', 'utf-8'))
        plt.show()

        # plot coefficient progression
        m_log_alphas = -np.log10(model.alphas_)
        #ax = plt.gca()
        plt.plot(m_log_alphas, model.coef_path_.T)
        plt.axvline(-np.log10(model.alpha_), linestyle='-', color='k',label=unicode('alpha CV', 'utf-8'))
        plt.ylabel(unicode('Regression Coefficients', 'utf-8'))
        plt.legend()
        plt.xlabel(unicode('-log(alpha)', 'utf-8'))
        plt.title(unicode('Regression Coefficients Progression for Lasso Paths', 'utf-8'))
        plt.show()        
        
            
        back = raw_input("\nEnter 'Y' for going back to main menu or else for exit: ")
        if(back=='Y'or back=='y'):
            main()
        else:
            print(B+"\nBye Bye!"+W)
                
    elif(choice == 4):
        print (P+"\n\t\t   ========================================================================================\n\t\t\t\t\t\t\tRANDOM FOREST\n\t\t   ========================================================================================"+W)
        feature_importances(X_train,Y_train,data_clean,forestModel)
        print (P+"\n\t\t   ========================================================================================\n\t\t\t\t\t\t       GRADIENT BOOSTING\n\t\t   ========================================================================================"+W)
        feature_importances(X_train,Y_train,data_clean,boostingModel)
        back = raw_input("\nEnter 'Y' for going back to main menu or else for exit: ")
        if(back=='Y'or back=='y'):
           main()
        else:
            print(B+"\nBye Bye!"+W)
    
    elif(choice == 5):        
        features = X_train.columns            
        def evaluate_param(parameter, num_range, index):
            grid_search = GridSearchCV(forestModel, param_grid = {parameter: num_range})
            grid_search.fit(X_train[features], Y_train)
    
            df = {}
            for i, score in enumerate(grid_search.grid_scores_):
                df[score[0][parameter]] = score[1]
        
            df = pd.DataFrame.from_dict(df, orient='index')
            df.reset_index(level=0, inplace=True)
            df = df.sort('index')
 
            plt.subplot(3,2,index)
            plot = plt.plot(df['index'], df[0])
            plt.title(parameter)
            return plot, df            
        
        # parameters and ranges to plot
        """
        param_grid = {"n_estimators": np.arange(2, 300, 2),
              "max_depth": np.arange(1, 28, 1),
              "min_samples_split": np.arange(1,150,1),
              "min_samples_leaf": np.arange(1,60,1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}        
        """
        param_grid = {"n_estimators": [7,8,10],
              "max_depth": [7,9],
              "min_samples_split": [8,10],
              "min_samples_leaf": [9,10],
              "max_leaf_nodes": [40,60],
              "min_weight_fraction_leaf": [0.1]}
        index = 1
        plt.figure(figsize=(16,12))
        for parameter, param_range in dict.items(param_grid):
            print "Plot ",index," Prepared..."
            evaluate_param(parameter, param_range, index)
            index += 1
        plt.show() 
            
        # Utility function to report best scores
        def report(grid_scores, n_top):
            top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
            for i, score in enumerate(top_scores):
                print("Model with rank: {0}".format(i + 1))
                print("Mean validation score: {0:.4f})".format(score.mean_validation_score,np.std(score.cv_validation_scores)))
                print("Parameters: {0}".format(score.parameters))
                print("")
            
        grid_search = GridSearchCV(forestModel, param_grid=param_grid)
        grid_search.fit(X_train[features], Y_train)

        report(grid_search.grid_scores_, 4)

        back = raw_input("\nEnter 'Y' for going back to main menu or else for exit: ")
        if(back=='Y'or back=='y'):
           main()
        else:
            print(B+"\nBye Bye!"+W)
    elif (choice==6):
        common_args = {'max_depth': 3, 'n_estimators': 500, 'subsample': 0.9, 'random_state': 2}

        models = [('learning rate: 1', GradientBoostingClassifier(learning_rate=1, **common_args)),
                  ('learning rate: 0.1', GradientBoostingClassifier(learning_rate=0.1, **common_args)),
                  ('learning rate: 0.01', GradientBoostingClassifier(learning_rate=0.01, **common_args))]        
        stage_preds = {}
        final_preds = {}
        
        count = 1
        for mname, m in models:
            print "Model ",count," Training..."
            m.fit(X_train, Y_train)
            stage_preds[mname] = {'X_train': list(m.staged_predict_proba(X_train)),  'X_test': list(m.staged_predict_proba(X_test))}
            final_preds[mname] = {'X_train': m.predict_proba(X_train),  'X_test': m.predict_proba(X_test)}
            count+=1
            
        def frame(i=0, log=False):
            for mname, _ in models:
                  plt.hist(stage_preds[mname]['X_train'][i][:,1], bins=np.arange(0,1.01,0.01), label=mname, log=log)
            plt.xlim(0,1)
            plt.ylim(0,8000)
            if log:
                plt.ylim(0.8,10000)
                plt.yscale('symlog')
                plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter())
            plt.legend(loc='upper center')
            return
        count = 1
        plt.figure(figsize=(16,10))
        for pos, fnum in enumerate((1, 5, 50, 100), 0):
            print "Plot ",count," Prepared..."
            plt.subplot2grid((3,2), (pos/2, pos%2))
            frame(fnum-1, True)
            plt.title("Predictions for each model at tree #%d (y axis on log scale)" % fnum)
            count+=1
            
        plt.subplot2grid((3,2), (2,0), colspan=2)
        plt.title("Final predictions for each model (y axis on linear scale)")
        frame(-1, False)
        plt.ylim(0,7000)
        plt.show()
        
        back = raw_input("\nEnter 'Y' for going back to main menu or else for exit: ")
        if(back=='Y'or back=='y'):
           main()
        else:
            print(B+"\nBye Bye!"+W)

def feature_extraction():
    count = 0
    clean_train_surveys = []
    csvfile = open("survey_results_public_.csv")
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        for i in range(19):
            if(not isinstance(row[i], str)):
                str(row[i])
        if(count != 0):
            clean_train_surveys.append(str(row))      
        count+=1
    
    vectorizer = CountVectorizer(analyzer='word',min_df=200,ngram_range=(1,1))
    train_data_features = vectorizer.fit_transform(clean_train_surveys)
    train_data_features = train_data_features.toarray()
    print "\nNumber of Features extracted: ",train_data_features.shape[1]
    vocab = vectorizer.get_feature_names()
    #print "\Features extracted: ", vocab
    
    dist = np.sum(train_data_features, axis=0)
    print "\nFeatures extracted with their respective count: \n"
    a = 1;
    for tag, count in zip(vocab, dist):
        print "feature",a,":",tag,"count: ", count
        a+=1
        
def menu():   
    #display title    
    print(P+"\n==================================================\nPROGRAMMER SALARY PREDICTION WITH MACHINE LEARNING\n==================================================\n"+W)
    print "\n1.Train and Save Models\n2.Compare Models (Lasso Regression, Random Forest, Gradient Boosting)\n3.Lasso Regression with different alphas",
    print "\n4.Feature Importances with Random Forest/Gradient Boosting\n5.Random Forest with Parameter Tuning\n6.Gradient Boosting with Learning Rate Tuning\n7.Feature Extraction with CountVectorizer (DEMO ONLY)\n8.Exit"
    choice = input("\nEnter your choice: ")
    
    return choice

def main():
    #print "\n" * 100
    choice = menu()
    
    if(choice == 1):
        trainAndSaveModels() 
        print "\nAll three models have been trained and the relavant data have been saved!!!"
        back = raw_input("\nEnter 'Y' for going back to main menu or else for exit: ")
        if(back=='Y'or back=='y'):
           main()
        else:
            print(B+"\nBye Bye!"+W)
    elif(choice == 8):
        print(B+"\nBye Bye!"+W)
        try:
            sys.exit(0)
        except:
            pass
    elif(choice == 7):
        feature_extraction()
        back = raw_input("\nEnter 'Y' for going back to main menu or else for exit: ")
        if(back=='Y'or back=='y'):
           main()
        else:
            print(B+"\nBye Bye!"+W)
    else:
        initialization(choice)

main()
    
                                                          



