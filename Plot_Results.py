import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable



def plot_results_learnperc():
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 4, 5, 8, 9]
    Algorithm = ['TERMS', 'PSO', 'GWO', 'WOA', 'RDA', 'PROPOSED']
    Classifier = ['TERMS', 'DENSENET', 'MOBILENET', 'VGG16', 'RESNET','INCEPTION','PROPOSED']


    value = Eval[4, :, 4:]
    value[:, :-1] = value[:, :-1] * 100
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, :])
    print('-------------------------------------------------- Algorithm Comparison - ',
          'Learning Percentage --------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
    print('-------------------------------------------------- Classifier Comparison - ',
          'Learning Percentage --------------------------------------------------')
    print(Table)

    Eval = np.load('Eval_all.npy', allow_pickle=True)
    learnper = [35, 55, 65, 75, 85]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100

        plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                 label="PSO-EHR-TM")
        plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label="GWO-EHR-TM")
        plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label="WOA-EHR-TM")
        plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                 label="RDA-EHR-TM")
        plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label="MMCS-RDA-EHR-TM")
        plt.xlabel('Learning Percentage (%)')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=4)
        path1 = "./Results/%s_line_1.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()



        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="DenseNet")
        ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="Mobilenet")
        ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="VGG16")
        ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="Resnet")
        ax.bar(X + 0.40, Graph[:, 9], color='y', width=0.10, label="Inception")
        ax.bar(X + 0.50, Graph[:, 10], color='k', width=0.10, label="MMCS-RDA-EHR-TM")
        plt.xticks(X + 0.25, ('35','55', '65', '75', '85'))
        plt.xlabel('Learning Percentage (%)')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/%s_bar_1.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

def plot_results_kfold():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 4, 5, 8, 9]
    Algorithm = ['TERMS', 'PSO', 'GWO', 'WOA', 'RDA', 'PROPOSED']
    Classifier = ['TERMS', 'DENSENET', 'MOBILENET', 'VGG16', 'RESNET','INCEPTION','PROPOSED']
    # for i in range(eval.shape[0]):
    value = eval[4, :, 4:]
    value[:, :-1] = value[:, :-1] * 100

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, :])
    print('-------------------------------------------------- ''Algorithm Comparison - ',
          'K - Fold --------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
    print('-------------------------------------------------- ''Classifier Comparison - ',
          'K - Fold --------------------------------------------------')
    print(Table)

    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    learnper = [1, 2, 3, 4, 5]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        # Graph = np.zeros(eval.shape[1:3])
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = eval[k, l, Graph_Term[j] + 4] * 100

        plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                 label="PSO-EHR-TM")
        plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label="GWO-EHR-TM")
        plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label="WOA-EHR-TM")
        plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                 label="RDA-EHR-TM")
        plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label="MMCS-RDA-EHR-TM")
        plt.xlabel('K - Fold')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=4)
        # "./Results/%s_line_1.png" % (Terms[Graph_Term[j]])
        path1 = "./Results/%s_line_2.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="DenseNet")
        ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="Mobilenet")
        ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="VGG16")
        ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="Resnet")
        ax.bar(X + 0.40, Graph[:, 9], color='y', width=0.10, label="Inception")
        ax.bar(X + 0.50, Graph[:, 10], color='k', width=0.10, label="MMCS-RDA-EHR-TM")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('K - Fold')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc=1)
        path1 = "./Results/%s_bar_2.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

def plot_Comp():
    matplotlib.use('TkAgg')
    Eval = np.load('Eval_all2.npy', allow_pickle=True)
    Eval1 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Classifier = ['TERMS', 'InSiNet[41]', 'F-SegNet[42]', 'U-RP-Net[43]', 'DTP-Net[46]','MMCS-RDA-EHR-TM']
    value = Eval[ 4, :, 4:]
    value[:, :-1] = value[:, :-1] * 100
    value1 = Eval1[ 4, :, 4:]
    value1[:, :-1] = value1[:, :-1] * 100
    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)

    Table.add_column(Classifier[len(Classifier) - 5], value[0, :])
    Table.add_column(Classifier[len(Classifier) - 4], value[2, :])
    Table.add_column(Classifier[len(Classifier) -3], value[1, :])
    Table.add_column(Classifier[len(Classifier) - 2], value[3, :])
    Table.add_column(Classifier[len(Classifier) - 1], value1[10, :])

    print('-----------------------------------Method Comparison--------------------------------------------------')
    print(Table)


def Statistical_():
    conv = np.load('Fitness.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['PSO-EHR-TM', 'GWO-EHR-TM', 'WOA-EHR-TM', 'RDA-EHR-TM', 'MMCS-RDA-EHR-TM']

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('--------------------------------------------------Statistical Analysis--------------------------------------------------')
    print(Table)

    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
             label="PSO-EHR-TM")
    plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
             label="GWO-EHR-TM")
    plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
             label="WOA-EHR-TM")
    plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
             label="RDA-EHR-TM")
    plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
             label="MMCS-RDA-EHR-TM")
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path1 = "./Results/conv.png"
    plt.savefig(path1)
    plt.show()

if __name__ == '__main__':
    plot_results_learnperc()
    plot_results_kfold()
    plot_Comp()
    Statistical_()
