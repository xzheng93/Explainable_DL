import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
from my_utils import load_data, plot_cm, my_plot_ana, plot_roc
import shap
import tensorflow as tf
# evaluate the model and calculate the SHAP

if __name__ == '__main__':
    model_name = 'cnn'
    plot_save_path = './plot'
    if model_name == 'cnn':
        data_path = './data/o_stride_data_nor.mat'
        model_path = './result/models/cnn.h5'
        window_size = 128
        bg_num = 1000
        fig_name = 'one_stride_anan.pdf'
        # SHAP explainer
        shap.explainers._deep.deep_tf.op_handlers[
            "AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
        shap.explainers._deep.deep_tf.op_handlers[
            "FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.

    elif model_name == 'gru':
        data_path = './data/e_stride_data_nor.mat'
        model_path = './result/models/gru.h5'
        window_size = 1024
        bg_num = 800
        fig_name = 'eight_strides_anan.pdf'
        tf.compat.v1.disable_v2_behavior()

    # load data
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(data_path, 0.2, 0.2)

    # load model
    model = keras.models.load_model(model_path)

    # evaluation
    pred_probability = model.predict(X_test)
    pred_probability_ = pred_probability[:, 1]
    pred_class = np.argmax(pred_probability, axis=1)
    true_class = np.argmax(y_test, axis=1)

    pre = metrics.precision_score(true_class, pred_class)
    rec = metrics.recall_score(true_class, pred_class)
    f1 = metrics.f1_score(true_class, pred_class)
    acc = metrics.accuracy_score(true_class, pred_class)
    fpr, tpr, thresholds = metrics.roc_curve(true_class, pred_probability_)
    auc = metrics.auc(fpr, tpr)

    print(f'Precision: {pre}')
    print(f'Recall: {rec}')
    print(f'f1_score: {f1}')
    print(f'Testing Accuracy: {acc}')
    print(f'AUC : {auc}')

    # plot roc
    auc = plot_roc(fpr, tpr, f'The ROC and AUC for {model_name}')
    plt.savefig(f'{plot_save_path}/ROC_{model_name}_shap.pdf', format='pdf')
    plt.show()

    # plot cm
    cm = metrics.confusion_matrix(true_class, pred_class)
    label = ['Adult', 'Older Adult']
    plot_cm(cm, label, f'Confusion Matrix')
    plt.savefig(f'{plot_save_path}/Confusion Matrix {model_name}_shap.pdf', format='pdf')
    plt.show()

    # select a set of background examples to take an expectation over
    background = X_train[np.random.choice(X_train.shape[0], bg_num, replace=False)]
    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)
    analysis = e.shap_values(X_test)
    analysis = analysis[0]
    np.save(f'./result/{model_path[16:-3]}.npy', analysis)

    my_plot_ana(analysis, X_test, window_size)
    plt.savefig(f'./Logs/{model_path[16:-3]}.pdf')
    plt.show()

