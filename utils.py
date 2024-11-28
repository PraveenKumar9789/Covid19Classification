import math

import matplotlib
import prettytable
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sbn

matplotlib.use('Qt5Agg')
prop = fm.FontProperties(fname='assets/FiraCode NF.ttf')
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['font.size'] = 14
plt.rcParams["figure.figsize"] = [19, 9]

CLASSES = ['Negative', 'Positive']


def reset_random():
    import os
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import warnings
    warnings.filterwarnings('ignore', category=Warning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf
    tf.compat.v1.random.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)


def print_performance_measures(y, pred, prob, labels):
    import os
    os.makedirs('results', exist_ok=True)
    measures = ['Accuracy Score', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC Score']
    class_report = {m: None for m in measures}
    save_path = 'results/confusion_matrix.png'
    title = 'Confusion Matrix'
    cm = confusion_matrix(y, pred)
    print('[INFO] {0}'.format(title))
    print(print_confusion_matrix(cm, labels))
    print('[INFO] Plotting Confusion Matrix')
    plot_conf_matrix(cm, title, labels=labels, path=save_path)
    print('[INFO] Plotting ROC Curve')
    save_path = 'results/roc_curve.png'
    title = 'ROC Curve'
    plot_roc_curve(y, prob, title, labels=labels, path=save_path)
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    class_report[measures[0]] = round((tp + tn) / (tp + tn + fp + fn), 4)
    class_report[measures[1]] = round(tp / (tp + fp), 4)
    class_report[measures[2]] = round(tp / (tp + fn), 4)
    class_report[measures[3]] = round(tn / (tn + fp), 4)
    n = (2 * class_report[measures[1]] * class_report[measures[2]])
    d = class_report[measures[1]] + class_report[measures[2]]
    class_report[measures[4]] = round(n / d, 4)
    class_report[measures[5]] = round(roc_auc_score(y, pred), 4)
    p_table = prettytable.PrettyTable(field_names=measures)
    for f in measures:
        p_table.align[f] = 'r'
    p_table.hrules = True
    p_table.add_row(class_report.values())
    print('[INFO] Classification Report')
    print('\n'.join(['\t{0}'.format(p_) for p_ in p_table.get_string().splitlines(keepends=False)]))
    with open('results/performance_measures.csv', 'w') as f:
        f.write(p_table.get_csv_string())
    return class_report


def print_confusion_matrix(cm, labels):
    p_table = prettytable.PrettyTable(['Class'] + labels + ['Total'])
    p_table.align['Class'] = 'l'
    for i, d in enumerate(cm):
        row = [labels[i]]
        row.extend([str(d_).center(8) for d_ in d])
        row.append(d.sum())
        p_table.add_row(row)
    p_table.add_row(['Total'] + cm.sum(axis=0).tolist() + [cm.sum()])
    p_table.hrules = True
    table_string = p_table.get_string().split('\n')
    table_string = [s.replace('-', '\u2500') for s in table_string]
    table_string = [s.replace('|', '\u2502') for s in table_string]
    table_string = [s[:-1] for s in table_string][:-1]
    table_string[-1] = table_string[-1].replace('│', ' ')
    rep = [['\u250C', '\u2510'],
           ['\u251C', '\u2524'],
           ['\u2514', '\u2518']]
    end_idx = table_string[0].rfind('+')
    for r in range(0, len(table_string) - 1, 2):
        extra_space = ' ' * (len(table_string[r]) - end_idx)
        table_string[r] = table_string[r][:end_idx] + extra_space
        if r == 0:
            table_string[r] = rep[0][0] + table_string[r][1:end_idx] + rep[0][1] + extra_space
            table_string[r] = table_string[r].replace('+', '\u252C')
        elif r == len(table_string) - 2:
            table_string[r] = rep[2][0] + table_string[r][1:end_idx] + rep[2][1] + extra_space
            table_string[r] = table_string[r].replace('+', '\u2534')
        else:
            table_string[r] = rep[1][0] + table_string[r][1:end_idx] + rep[1][1] + extra_space
            table_string[r] = table_string[r].replace('+', '\u253C')
    table_string.insert(0, 'Predicted'.center(end_idx))
    add_idx = math.ceil((len(table_string) - 2) / 2)
    for i, s in enumerate(table_string):
        if i == add_idx:
            table_string[i] = 'Actual ' + s
        else:
            table_string[i] = ' ' * 7 + s
    return '\n'.join(['\t\t{0}'.format(t) for t in table_string])


def plot_conf_matrix(conf_mat, title, labels, path):
    fig = plt.figure(num=1, figsize=(19, 9))
    sbn.heatmap(conf_mat, annot=True, cmap='YlGnBu', annot_kws={"size": 15}, linewidths=0.5, fmt='d',
                yticklabels=labels, xticklabels=labels)
    plt.xlabel('Predicted Class', labelpad=15)
    plt.ylabel('Actual Class', labelpad=15)
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.close(fig)


def plot_roc_curve(y, prob, title, labels, path):
    n_classes = len(labels)
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y, prob[:, i],
                                      pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    y_true = label_binarize(y, classes=range(n_classes))
    y_true = np.hstack((1 - y_true, y_true))
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(),
                                              prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    fig, ax = plt.subplots(1, 1, num=1)
    plt.title(title)
    for i in range(n_classes):
        color = plt.cm.get_cmap('Dark2')(float(i) / n_classes)
        ax.plot(fpr[i], tpr[i], lw=2, color=color,
                label='{0} (area = {1:0.2f})'
                      ''.format(labels[i], roc_auc[i]))
    ax.plot(fpr['micro'], tpr['micro'],
            label='Micro-Average '
                  '(area = {0:0.2f})'.format(roc_auc['micro']),
            color='deeppink', linestyle=':', linewidth=2)
    ax.plot(fpr['macro'], tpr['macro'],
            label='Macro-Average '
                  '(area = {0:0.2f})'.format(roc_auc['macro']),
            color='navy', linestyle=':', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Threshold == 0.5')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.tick_params()
    ax.legend(loc='lower right')
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.close(fig)
