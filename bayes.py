from math import log
import itertools
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix

# dataset: Https://Github.Com/Idontflow/Olidhttps://github.com/idontflow/OLID
# https://paperswithcode.com/paper/predicting-the-type-and-target-of-offensive


HOME_DIR = "/csse/users/grh102/Documents/cosc442/cosc442-supervised-systems-master/OLID/"


def train_naive_bayes(D, C):
    """
    train_naive_bayes: given documents and a list of classes,
    calculates probability of classes occurring, and probability
    of a word given a class
    
    Input:
        D: list(str). lists of all documents
        C: list(any). list of unique classes
    
    Returns:
        log_prior: dict: class -> P(class). Log probability of a class
        log_likelihood: dict: (word, class): P(word | class). Log probability of a 
            word occurring in a given class
        V: set(str): vocabulary of all words in documents
        C: list(any): list of unique classes
    """
    log_prior = dict()   # class: P(c)
    log_likelihood = dict()   # (word, class): P(word | class)
    
    V = set().union(*[set(d.split()) for d, _ in D])  # form a set of all words in docs
    
    n_doc = len(D)
      
    for class_ in C:
        c_documents = [document for document, c in D if c == class_]
        n_c = len(c_documents)
        
        log_prior[class_] = log(n_c / n_doc)
        
        bigdoc = []
        for doc in c_documents:
            bigdoc += doc
        
        counter = Counter(bigdoc)
        
        denominator = sum([counter[w] + 1 for w in V])
        
        for w in V:
            count_wc = counter[w] + 1
            log_likelihood[(w, class_)] = log(count_wc / denominator)
            
    return log_prior, log_likelihood, V, C


def test_naive_bayes(test_doc, log_prior, log_likelihood, C, V):
    """
    test_naive_bayes: given a document and the bayesian probabilies/vocab,
    return the most likely class
    
    Inputs:
        test_doc: str, document to test
        log_prior: dict: class -> P(class). Log probability of a class
        log_likelihood: dict: (word, class): P(word | class). Log probability of a 
            word occurring in a given class
        V: set(str): vocabulary of all words in documents
        C: list(any): list of unique classes
        
    Returns:
        class: any, most likely class
    """
    
    sum_classes = dict()   # class: log probability of that class
    for class_ in C:
        sum_classes[class_] = log_prior[class_]
        
        for word in test_doc.split():
            if word in V:
                sum_classes[class_] += log_likelihood[(word, class_)]
                
    return max(sum_classes, key=sum_classes.get)


def train_bayes():
    """
    train_bayes: read the tsv of training data and 
    calculates the probability of each class, and 
    each class given the presence of a word
    
    Returns:
        log_prior: dict: class -> P(class). Log probability of a class
        log_likelihood: dict: (word, class): P(word | class). Log probability of a 
            word occurring in a given class
        V: set(str): vocabulary of all words in documents
        C: list(any): list of unique classes
    """
    train = pd.read_csv(HOME_DIR + 'olid-training-v1.0.tsv', sep='\t')
    
    # convert values into list to pass into training function
    documents = list(train.tweet.values)
    labels = list(train.subtask_a.values)
    C = list(set(labels))
    
    return train_naive_bayes(list(zip(documents, labels)), C)


def f1(tp, fp, fn):
    return tp / (tp + 0.5 * (fp + fn))


def print_results(test_labels, preds):
    """
    print_results: print results from predicted labels vs gold standard labels
    
    Inputs:
        test_labels: list(any), gold standard labels to test against
        preds: list(any), labels predicted by the model
    """
    results = confusion_matrix(test_labels, preds)
    tn, fp, fn, tp = results.ravel()
    print("Accuracy: {:.4f}".format((tn + tp) / len(preds)))
    print("Confusion matrix: \n", results)
    print("F1: {:.4f}".format(f1(tp, fp, fn)))


def evaluate(log_prior, log_likelihood, V, C):
    """
    evaluate: evaluate all of the test documents from the csv's and print the results
    
    Inputs:
        log_prior: dict: class -> P(class). Log probability of a class
        log_likelihood: dict: (word, class): P(word | class). Log probability of a 
            word occurring in a given class
        V: set(str): vocabulary of all words in documents
        C: list(any): list of unique classes
    """
    test_labels_df = pd.read_csv(HOME_DIR + 'labels-levela.csv', sep='\t', header=None, names=["id_label"])
    test_labels = [val.split(",")[1] for val in test_labels_df.id_label.values]
    
    test_docs_df = pd.read_csv(HOME_DIR + 'testset-levela.tsv', sep='\t')
    test_docs = list(test_docs_df.tweet.values)
    
    preds = []
    for d, c in zip(test_docs, test_labels):
        preds.append(test_naive_bayes(d, log_prior, log_likelihood, C, V))
        
    print_results(test_labels, preds)
    

if __name__ == "__main__":
    log_prior, log_likelihood, V, C = train_bayes()
    evaluate(log_prior, log_likelihood, V, C)
    

