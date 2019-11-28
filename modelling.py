from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from evaluation import plot_confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt

def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True, \
                 print_cm=True, cm_cmap=plt.cm.Greens):
    
    
    # to store results at various phases
    results = dict()
    
    # time at which model starts training 
    train_start_time = datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done \n \n')
    train_end_time = datetime.now()
    results['training_time'] =  train_end_time - train_start_time
    print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))
    # predict test data
    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
    
    # calculate overall accuracty of the model
    print('---------Performance Score--------------')
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision=precision_score(y_true=y_test, y_pred=y_pred,average='weighted')
    recall=recall_score(y_true=y_test, y_pred=y_pred,average='weighted')
    weighted_f1=f1_score(y_true=y_test, y_pred=y_pred,average='weighted')
    print('\n Weighted F1:{}'.format(micro_f1))
    print('\n Precision:{}'.format(precision))
    print('\n Recall:{}'.format(recall))
    print('\n Accuracy:{}\n'.format(accuracy))
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    plt.figure(figsize=(8,8))
    plt.grid(b=False)
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
    plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classifiction Report |')
    clf_report = classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = clf_report
    print(clf_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return results,model
