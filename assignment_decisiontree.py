import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # we need this to split the dataset
from sklearn.feature_extraction.text import CountVectorizer # we need this to generate a vector matrix from the words in the messages
import re # this is the regular expression library
import nltk # this is the natural language toolkit library
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn

#uncomment to download nltk stuff**
#import ssl
#try:
    #_create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
    #pass
#else:
    #ssl._create_default_https_context = _create_unverified_https_context
#nltk.download('punkt_tab')
#nltk.download('wordnet')
#nltk.download('stopwords')

#start
#import data
data = pd.read_csv("cleaned_spam.csv")

#clean data digitally
#remove duplicates
data = data.drop_duplicates(subset=['v2'])

#see data info
print("Data overview:")
print(data.head())
print()
print("Data information:")
print(data.info())
print()
print("Data description:")
print(data.describe())
print()

#remove links from data
no_link = [re.sub(r'http\S+', '', i) for i in data["v2"]]

#remove symbols from data
clean = [re.sub('[^a-zA-Z0-9 ]', '', i) for i in no_link]

#change data to lowercase
lower = [i.lower() for i in clean]

#natural language toolkit cleaning
#to make the text be able to be analyzed by the models
tokens = [nltk.word_tokenize(w) for w in lower]
lemma = WordNetLemmatizer()
lemmatized = [[lemma.lemmatize(w) for w in text] for text in tokens]
without_stopwords = [[w for w in text if w not in stopwords.words('english')] for text in lemmatized]
vectorizer = CountVectorizer(max_features=20000)
X = vectorizer.fit_transform([' '.join(text) for text in without_stopwords]).toarray()

#data splitting, training and testing
y = np.array(data['v1'])
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.3)
model = DecisionTreeClassifier()
model_name = "Decision Tree"
model.fit(X_train,y_train)
model.score(X_test,y_test)
Predictions = model.predict(X_test)

#model evaluation
Score = accuracy_score(y_test, Predictions)
print(f"{model_name} Accuracy: {Score:.2f} \n")
class_report = classification_report(y_test, Predictions)
print(model_name, "Report:\n", class_report)

y_pred = model.predict(X_test)
conf = confusion_matrix(y_true=y_test, y_pred=y_pred)

labels = ["ham", "spam"]
disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"{model_name} Confusion Matrix")
plt.show()

#seaborn.heatmap(conf,annot=True,fmt=".1f",linewidths=0.5)
#plt.show()