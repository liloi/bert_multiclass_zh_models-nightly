import pdb
import pickle
import matplotlib.pyplot as plt
import matplotlib.style as style

class MetricPlot(object):
    def __init__(self, file_pickle_path):
        self.file_pickle = file_pickle_path

    def draw_all_curves(self):
        with open(self.file_pickle + '/metric/metric.pickle', 'rb') as f:
            metric_data = pickle.load(f)
            reports = metric_data['reports']
            style.use("bmh")
            plt.figure(figsize=(8, 12))

            class_list = []
            for report in reports:
                class_list.extend(list(report.keys()))
            class_list = list(set(class_list))
            class_list.remove('accuracy')
            class_list.remove('macro avg')
            class_list.remove('weighted avg')
            class_list = sorted(class_list)

            # 每个类别分别打印precision、recall、f1
            class_num = len(class_list)
            for c in range(class_num):
                plt.subplot(class_num, 1, c + 1)
                for m in {'precision', 'recall', 'f1-score'}:
                    plt.plot([report[str(c)][m] for report in reports],
                            label='Class {0} {1}'.format(c, m))
                    plt.legend(loc='lower right')
                    plt.ylabel('Class {}'.format(c))
                    plt.title('Class {} Curves'.format(c))
            plt.show()
            """
            plt.subplot(3, 1, 1)
            m = 'precision'
            for c in class_list:
                plt.plot([report[str(c)][m] for report in reports],
                        label='Class {0} {1}'.format(c, m))
            plt.legend(loc='lower right')
            plt.ylabel(m)
            plt.title('Validation {} Curve'.format(m))

            plt.subplot(3, 1, 2)
            m = 'recall'
            for c in class_list:
                plt.plot([report[str(c)][m] for report in reports],
                        label='Class {0} {1}'.format(c, m))
            plt.legend(loc='lower right')
            plt.ylabel(m)
            plt.title('Validation {} Curve'.format(m))

            plt.subplot(3, 1, 3)
            m = 'f1-score'
            for c in class_list:
                plt.plot([report[str(c)][m] for report in reports],
                        label='Class {0} {1}'.format(c, m))
            plt.legend(loc='lower right')
            plt.ylabel(m)
            plt.title('Validation {} Curve'.format(m))
            plt.show()
            """

    def draw_metric_curves(self):
        with open(self.file_pickle + '/metric/metric.pickle', 'rb') as f:
            metric_data = pickle.load(f)
            f1 = metric_data['f1']
            recall = metric_data['recall']
            precision = metric_data['precision']

            epochs = len(f1)

            style.use("bmh")
            plt.figure(figsize=(8, 12))

            plt.subplot(3, 1, 1)
            plt.plot(range(1, epochs+1), f1, label='Val F1')
            plt.legend(loc='lower right')
            plt.ylabel('F1')
            plt.title('Validation F1 Curve')

            plt.subplot(3, 1, 2)
            plt.plot(range(1, epochs+1), recall, label='Val Recall')
            plt.legend(loc='lower right')
            plt.ylabel('Recall')
            plt.title('Validation Recall Curve')

            plt.subplot(3, 1, 3)
            plt.plot(range(1, epochs+1), precision, label='Val Precision')
            plt.legend(loc='lower right')
            plt.ylabel('Precision')
            plt.title('Validation Precision Curve')
            plt.xlabel('epoch')
            plt.show()

    def draw_history_curves(self):
        """Plot the learning curves of loss and macro f1 score 
        for the training and validation datasets.

        Args:
            history: history callback of fitting a tensorflow keras model 
        """
        with open(self.file_pickle + '/history/hist.pickle', 'rb') as f:
            history = pickle.load(f)

            loss = history['loss']
            val_loss = history['val_loss']
            accuracy = history['test_accuracy']
            val_accuracy = history['val_test_accuracy']

            epochs = len(loss)

            style.use("bmh")
            plt.figure(figsize=(8, 8)) 

            plt.subplot(2, 1, 1)
            plt.plot(range(1, epochs+1), loss, label='Training Loss')
            plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')

            plt.subplot(2, 1, 2)
            plt.plot(range(1, epochs+1), accuracy, label='Training Accuracy')
            plt.plot(range(1, epochs+1), val_accuracy, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.show()

if __name__ == '__main__':
    c = MetricPlot('/home/work/liran05/xxx')
    c.draw_metric_curves()
