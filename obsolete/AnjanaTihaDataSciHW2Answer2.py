
# coding: utf-8

# In[2]:


#Spam filtering
import numpy as np
import pandas as pd
import os
import email
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix
from bs4 import BeautifulSoup
import re


#removing extranous characters 

def data_from_file():
    
    target = []
    index = []
    rows = []
    
    #importing non-spam folder's file
    
    flist = os.listdir("spamassasin\\ham")   
    
    for f in flist:

        ifile=open("spamassasin\\ham\\" + f, encoding = "ISO-8859-1")
        
        rawtext=""
        
        rawtext = file_read(ifile)
        
        msg = email.message_from_string(rawtext)
        subject = str(msg['Subject'])
        
        body = email_parse_subject_body(rawtext)
        
        subjectandbody=subject + "\n" + body
        
        rows.append({'text': subjectandbody, 'class': 0})
        index.append(f)

        
    #importing spam folder's file
    
    flist = os.listdir("spamassasin\\spam")

    for f in flist:

        ifile=open("spamassasin\\spam\\" + f, encoding = "ISO-8859-1")
       
        rawtext=""
        
        rawtext = file_read(ifile)
        
        msg = email.message_from_string(rawtext)
        subject = str(msg['Subject'])
        
        body = email_parse_subject_body(rawtext)
        
        subjectandbody = subject + "\n" + body
        
        rows.append({'text': subjectandbody, 'class': 1})
        index.append(f)

    data_frame_from_email_and_class = pd.DataFrame(rows, index=index)
    return data_frame_from_email_and_class



#file read function
def file_read(ifile):
    
    rawtext = ""
    lines = ifile.readlines()
    for l in lines:
        rawtext = rawtext + l
    ifile.close()
    return rawtext



#extracting subject and body

def email_parse_subject_body(rawtext):
    
    emailText = email.message_from_string(rawtext)
    maintype = emailText.get_content_maintype()
    
    if maintype == 'text':
        cleanedemail = emailText.get_payload()  
    else:
        cleanedemail = ""
    preprocessed_email=preprocessing(cleanedemail)
    
    return preprocessed_email

    
#Preprocessing
def preprocessing(htmltext):
    
    soup = BeautifulSoup(htmltext,"lxml")
    
    for script in soup(["script", "style"]):
        script.extract()    
        
    email_text = soup.get_text()
    lines = (line.strip() for line in email_text.splitlines())
    
    group = (phrase.strip() for line in lines for phrase in line.split("  "))
    email_text = '\n'.join(group for group in group if group)
    
    sp_character = re.compile('(<|>|^|&|||_|-)')
    sp_character_removed = sp_character.sub('', email_text)
    
    return sp_character_removed


#Training and Testing 5 folds
def model_performance(data_frame, targets, vectorizer, classifier):

    precision_list = []
    recall_list = []
    df1_list = []
    
    #Folding data in K folds maintaining balanced spam and non-spam emails in training
    skf = StratifiedKFold(targets, n_folds=5, shuffle = True)

    for train_index, test_index in skf:

        X_train, X_test = data_frame['text'][train_index].values, data_frame['text'][test_index].values
        y_train, y_test = targets[train_index], targets[test_index]

        X_vect = vectorizer.fit_transform(X_train)
        classifier.fit(X_vect, y_train)
        X_text_vect = vectorizer.transform(X_test)
        y_predict = classifier.predict(X_text_vect)
        
        confusion = confusion_matrix(y_test, y_predict)
        
        precision_value = precision_func(confusion)
        precision_list.append(precision_value)
        
        recall_value = recall_func(confusion)
        recall_list.append(recall_value)
        
        df1_value = df1_func(precision_value, recall_value)
        df1_list.append(df1_value)
    
    avgscore(precision_list, recall_list, df1_list)
    stdscore(precision_list, recall_list, df1_list)


#Average score for precision, recall and DF-1 calculation

def avgscore(precision_list, recall_list, df1_list):
    
    avg_precision = np.mean(precision_list, axis = 0)
    avg_recall = np.mean(recall_list, axis = 0)
    avg_df1 = np.mean(df1_list, axis = 0)
    
    print("Average score of 5 folds \n Precision:  ", avg_precision, "  Recall:  ", avg_recall, " DF-1: ", avg_df1, "\n")
    
    
#Standard deviation score for precision, recall and DF-1 calculation

def stdscore(precision_list, recall_list, df1_list):
    
    std_precision = np.std(precision_list, axis = 0)
    std_recall = np.std(recall_list, axis = 0)
    std_df1 = np.std(df1_list, axis = 0)
    
    print("Standard Deviation of score of 5 folds :\n", "Precision:  ", std_precision, "  Recall:  ", std_recall, "  DF-1: ", std_df1, "\n")

    
#precision calculation
def precision_func(confusion):  
   
    precision = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    return precision


#recall calculation
def recall_func(confusion):
    
    recall = confusion[1][1] / (confusion[1][1] + confusion[1][0])
    return recall


#DF-1 calculation
def df1_func(precision, recall):
    
    df1 = 2 * precision * recall/(precision + recall)
    return df1


# Function call and rest
    
data_frame = data_from_file()
targets = data_frame['class'].values


count_vectorizer = CountVectorizer()
tfid_vectorizer = TfidfVectorizer()
naive_bayes=MultinomialNB()

LogisticRegression_L1_c_5=linear_model.LogisticRegression(penalty='l1', C = 0.5)
LogisticRegression_L1_c_1=linear_model.LogisticRegression(penalty='l1', C = 1.0)
LogisticRegression_L2_c_5=linear_model.LogisticRegression(penalty='l2', C = 0.5)
LogisticRegression_L2_c_1=linear_model.LogisticRegression(penalty='l2', C = 1.0)


print("Naive Bayes count vectorized ")
model_performance(data_frame, targets, count_vectorizer, naive_bayes)
print("\nNaive Bayes tfid vectorized \n")
model_performance(data_frame, targets, tfid_vectorizer, naive_bayes)
print("\nLogistic Regression count vectorized L1 regularization C = 0.5\n")
model_performance(data_frame, targets, count_vectorizer, LogisticRegression_L1_c_5)
print("\nLogistic Regression tfid vectorized L1 regularization C = 0.5\n")
model_performance(data_frame, targets, tfid_vectorizer, LogisticRegression_L1_c_5)
print("\nLogistic Regression count vectorized L1 regularization C = 1\n")
model_performance(data_frame, targets, count_vectorizer, LogisticRegression_L1_c_1)
print("\nLogistic Regression tfid vectorized L1 regularization C = 1\n")
model_performance(data_frame, targets, tfid_vectorizer, LogisticRegression_L1_c_1)
print("\nLogistic Regression count vectorized L2 regularization C = 0.5\n")
model_performance(data_frame, targets, count_vectorizer, LogisticRegression_L2_c_5)
print("\nLogistic Regression tfid vectorized L2 regularization C = 0.5\n")
model_performance(data_frame, targets, tfid_vectorizer, LogisticRegression_L2_c_5)
print("\nLogistic Regression count vectorized L2 regularization C = 1\n")
model_performance(data_frame, targets, count_vectorizer, LogisticRegression_L2_c_1)
print("\nLogistic Regression tfid vectorized L2 regularization C = 1\n")
model_performance(data_frame, targets, tfid_vectorizer, LogisticRegression_L2_c_1)






