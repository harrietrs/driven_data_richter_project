from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools


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

def plot_params(x_label, param_values, results, scoring, y_low_bound, y_high_bound):
    
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

    plt.xlabel(x_label)
    plt.ylabel("Score")

    ax = plt.gca()
#     ax.set_xlim(0, 20)
    ax.set_ylim(y_low_bound, y_high_bound)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(param_values.data, dtype=float)
    
    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s'
                                    % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))
    
        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]
    
        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
    
        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))
    
    plt.legend(loc="best")
    plt.grid(False)
    plt.show()