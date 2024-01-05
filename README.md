# Email-Spam-Filteration-Model

# PROBLEM STATEMENT
- The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

- The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

# PROJECT OVERVIEW:
-- First of all,we need to check whether it is regression task or classifiction task...
        
    It is a classification task as we have to classify whtere a specific mail is spma or not.
  -- Here,we are going to use Naive Bayes and Natural Language Processing(NLP) to get this task done.

  Natural Language Processing(NLP) enables computers to understand natural language as humans do. Whether the language is spoken or written, natural language processing uses artificial intelligence to take real-world input, process it, and make sense of it in a way a computer can understand.

  # STEPS TAKEN:

So,here are the step I followed in this problem statement:

--> Step-1: Importing the libraries and Loading the dataset.
-- Begin by importing essential libraries such as NumPy for basic calculations, Pandas for handling dataframes, Matplotlib for plotting graphs, and Seaborn for visualizing data. 
-- Load the dataset named "emails" from a CSV file using the Pandas library.

--> Step-2: Data Visualization
-- Separate the dataset into two parts: "spam_df" for spam emails and "ham" for non-spam emails.
-- Plot a histogram depicting the relationship between email length and frequency.
-- Calculate the percentage distribution of spam and non-spam emails.

--> Step-3 Creating testing and training dataset/data cleaning.
-- Initiate data cleaning by removing punctuation from the text, as it does not contribute significantly to the analysis.
-- Then Eliminate stopwords, common words in the English language that don't provide substantial information. Stop words are commonly used words in a language that are used in Natural Language Processing (NLP). They are used to remove words that are so common that they don't provide much useful information. Eg- A,An,The,On,Of,We,I.
-- Now by using count vectorizer, we convert a collection of text documents into a numerical representation. It is part of the scikit-learn library, a popular machine learning library in Python.

--> Step-4 Training the model 
-- First spilt the dataser into four parts: X_train,X_test,y_train,y_test.
-- Importing the Multinomial Naive bayes classifier for training purposes.
-- Then fit the classifier using our training set.

--> Step-5 Evaluating the model
-- First import the tools such as classification report and confusion matrix.
-- Now plot the confusion matrix for training and testing set.A confusion matrix is a table that summarizes the performance of a classification model. It's a performance evaluation tool in machine learning that displays the number of true positives, true negatives, false positives, and false negatives. 
-- Then make classification report of model on testing set.The classification report shows a representation of the main classification metrics on a per-class basis.The classification report visualizer displays the precision, recall, F1, and support scores for the model.

--> Step-6 Adding additional feature TF-IDF
-- Tf–idf stands for "Term Frequency–Inverse Document Frequency" is a numerical statistic used to reflect how important a word is to a document in a collection or corpus of documents. TFIDF is used as a weighting factor during text search processes and text mining.
-- For this, we use TfidfTransformer. First fit the training set into this classifier and then repeat all the process on new dataset.




