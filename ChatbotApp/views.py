from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pymysql
from numpy import dot
from numpy.linalg import norm
from django.core.files.storage import FileSystemStorage
from datetime import date
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from numpy import dot
from numpy.linalg import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

global uname, tfidf_vectorizer, scaler
global X_train, X_test, y_train, y_test
accuracy, precision, recall, fscore = [], [], [], []

dataset = pd.read_csv("Dataset/BankFAQs.csv")
dataset = dataset.dropna()
question = []
answer = []
Y = []
labels = np.unique(dataset['Class'])
le = LabelEncoder()
dataset['Class'] = pd.Series(le.fit_transform(dataset['Class'].astype(str))) #encoding non-numeric labels into numeric
Y = dataset['Class']
dataset = dataset.values
#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

question = np.load("model/question.npy")
answer = np.load("model/answer.npy")
Y = np.load("model/Y.npy")

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=600)
tfidf_X = tfidf_vectorizer.fit_transform(question).toarray()        
print(tfidf_X.shape)    

#scaler = StandardScaler()
#X = scaler.fit_transform(tfidf_X)
X = tfidf_X
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split data into train & test
X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1) #split data into train & test

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)
calculateMetrics("KNN", predict, y_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)

svm_cls = svm.SVC()
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", predict, y_test)

def TextChatbot(request):
    if request.method == 'GET':
        return render(request, 'TextChatbot.html', {})

def TrainML(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        global accuracy, precision, recall, fscore
        algorithms = ['KNN', 'Random Forest', 'SVM']
        for i in range(len(algorithms)):
            output+='<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td><td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)

def UpdateStatus(request):
    if request.method == 'GET':
        lid = request.GET['lid']
        status = request.GET['status']
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "update loan set status='"+status+"' where loan_id='"+lid+"'"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        context= {'data':'Application Status successfully Updated : '+status}
        return render(request, 'AdminScreen.html', context)
        

def ViewApplications(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Application ID</th><th><font size="" color="black">Applicant Name</th>'
        output+='<th><font size="" color="black">Loan Purpose</th><th><font size="" color="black">Amount</th><th><font size="" color="black">Pan No</th>'
        output+='<th><font size="" color="black">Aadhar No</th><th><font size="" color="black">Applied Date</th>'
        output+='<th><font size="" color="black">Status</th><th><font size="" color="black">Accept Application</th><th><font size="" color="black">Reject Application</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from loan where status='Pending'")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+str(row[0])+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td><td><font size="" color="black">'+row[4]+'</td>'
                output += '<td><font size="" color="black">'+row[5]+'</td>'
                output += '<td><font size="" color="black">'+row[6]+'</td><td><font size="" color="black">'+row[7]+'</td>'
                output+='<td><a href=\'UpdateStatus?lid='+str(row[0])+'&status=Accepted\'><font size=3 color=black>Click Here to Accept</font></a></td>'
                output+='<td><a href=\'UpdateStatus?lid='+str(row[0])+'&status=Rejected\'><font size=3 color=black>Click Here to Reject</font></a></td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)       

def ApplyLoan(request):
    if request.method == 'GET':
        return render(request, 'ApplyLoan.html', {})

def ApplyLoanAction(request):
    if request.method == 'POST':
        global uname
        purpose = request.POST.get('t1', False)
        amount = request.POST.get('t2', False)
        pan = request.POST.get('t3', False)
        aadhar = request.POST.get('t4', False)
        today = str(date.today())
        loan_id = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select max(loan_id) FROM loan")
            rows = cur.fetchall()
            for row in rows:
                loan_id = row[0]
        if loan_id is not None:
            loan_id += 1
        else:
            loan_id = 1
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO loan(loan_id,username,loan_purpose,amount,pan_no,aadhar_no,applied_date,status) VALUES('"+str(loan_id)+"','"+uname+"','"+purpose+"','"+amount+"','"+pan+"','"+aadhar+"','"+today+"','Pending')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        context= {'data':'Your Application submitted successfully with ID : '+str(loan_id)+"<br/>Our Admin will reply"}
        return render(request, 'ApplyLoan.html', context)

def saveInteraction(user_question, output):
    global uname
    today = str(date.today())
    interact_id = 0
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select max(interact_id) FROM interaction")
        rows = cur.fetchall()
        for row in rows:
            interact_id = row[0]
    if interact_id is not None:
        interact_id += 1
    else:
        interact_id = 1
    db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
    db_cursor = db_connection.cursor()
    student_sql_query = "INSERT INTO interaction(interact_id,username,question,answer,interact_date) VALUES('"+str(interact_id)+"','"+uname+"','"+user_question+"','"+output+"','"+today+"')"
    db_cursor.execute(student_sql_query)
    db_connection.commit()        

def ChatData(request):
    if request.method == 'GET':
        global answer, tfidf_vectorizer, X, question, scaler
        user_question = request.GET.get('mytext', False)
        query = user_question
        print(query)
        query = query.strip().lower()
        query = cleanText(query)#clean description 
        testData = tfidf_vectorizer.transform([query]).toarray()
        #testData = scaler.transform(testData)
        testData = testData[0]
        print(testData.shape)
        output =  "Sorry! unable to answer"
        index = -1
        max_accuracy = 0
        for i in range(len(X)):
            predict_score = dot(X[i], testData)/(norm(X[i])*norm(testData))
            if predict_score > max_accuracy:
                max_accuracy = predict_score
                index = i
        if index != -1:
            output = answer[index]
            saveInteraction(user_question, output)
        print(output)    
        return HttpResponse("Chatbot: "+output, content_type="text/plain")

def AddQuestion(request):
    if request.method == 'GET':
       return render(request, 'AddQuestion.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})
    
def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})    

def AdminLoginAction(request):
    if request.method == 'POST':
        global userid
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if user == "admin" and password == "admin":
            context= {'data':'Welcome '+user}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid Login'}
            return render(request, 'AdminLogin.html', context)

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)


def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Signup Process Completed. You can Login now"
        context= {'data': status}
        return render(request, 'Signup.html', context)

def ViewUser(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Username</th><th><font size="" color="black">Password</th><th><font size="" color="black">Contact No</th>'
        output+='<th><font size="" color="black">Email ID</th><th><font size="" color="black">Address</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from register")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+row[0]+'</td><td><font size="" color="black">'+str(row[1])+'</td><td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td><td><font size="" color="black">'+row[4]+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)    


def ViewChats(request):
    if request.method == 'GET':
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Interaction ID</th><th><font size="" color="black">Username</th><th><font size="" color="black">Question</th>'
        output+='<th><font size="" color="black">Chatbot Answer</th><th><font size="" color="black">Interaction Date</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from interaction")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+str(row[0])+'</td><td><font size="" color="black">'+str(row[1])+'</td><td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td><td><font size="" color="black">'+row[4]+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'AdminScreen.html', context)   


def ViewStatus(request):
    if request.method == 'GET':
        global uname
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Application ID</th><th><font size="" color="black">Applicant Name</th>'
        output+='<th><font size="" color="black">Loan Purpose</th><th><font size="" color="black">Amount</th><th><font size="" color="black">Pan No</th>'
        output+='<th><font size="" color="black">Aadhar No</th><th><font size="" color="black">Applied Date</th>'
        output+='<th><font size="" color="black">Status</th></tr>'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'BankChatbot',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * from loan where username='"+uname+"'")
            rows = cur.fetchall()
            output+='<tr>'
            for row in rows:
                output+='<td><font size="" color="black">'+str(row[0])+'</td><td><font size="" color="black">'+str(row[1])+'</td>'
                output+='<td><font size="" color="black">'+row[2]+'</td><td><font size="" color="black">'+row[3]+'</td><td><font size="" color="black">'+row[4]+'</td>'
                output += '<td><font size="" color="black">'+row[5]+'</td>'
                output += '<td><font size="" color="black">'+row[6]+'</td><td><font size="" color="black">'+row[7]+'</td></tr>'
        output+= "</table></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'UserScreen.html', context)    
