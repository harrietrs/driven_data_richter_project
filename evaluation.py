from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import random
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def measure_f1(model,x_test,y_test):
    y_pred=model.predict(x_test)
    micro_f1=f1_score(y_true=y_test, y_pred=y_pred,average='micro')
    print('\n Micro F1 Score of Test Data:{}'.format(micro_f1))
    

def get_random_score(data_values, data_labels):
    random.seed = 42
    y_pred_random=[]
    for i in range(data_values.shape[0]):
        y_pred_random.append(random.randint(1,3))
    score=f1_score(y_pred_random,data_labels,average='micro')
    print('F1 Score of random model on CV data is:',score)
    