#-*- encoding:utf-8 -*-
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

            class_list = []
            for report in reports:
                class_list.extend(list(report.keys()))
            class_list = list(set(class_list))
            class_list.remove('accuracy')
            class_list.remove('macro avg')
            class_list.remove('weighted avg')

            # 每个类别分别打印precision、recall、f1
            class_num = len(class_list)
            plt.figure(figsize=(8, class_num * 4))
            for i, v in enumerate(class_list):
                plt.subplot(class_num, 1, i + 1)
                for m in {'precision', 'recall', 'f1-score'}:
                    plt.plot([report[v][m] for report in reports],
                           label='Class {0} {1}'.format(v, m))
                    plt.legend(loc='lower right')
                    plt.ylabel('Class {}'.format(v))
                    plt.title('Class {} Curves'.format(v))
            plt.show()

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
        with open(self.file_pickle + '/metric/hist.pickle', 'rb') as f:
            history = pickle.load(f)

            loss = history['loss']
            val_loss = history['val_loss']
            accuracy = history['accuracy']
            val_accuracy = history['val_accuracy']

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
    c = MetricPlot('/home/disk2/liran05/github/tensorflow/models/official/nlp/bert/process_dir')
    c.draw_all_curves()
