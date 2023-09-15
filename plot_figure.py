import numpy as np
import matplotlib.pyplot as plt
from my_utils import load_data, my_interp, find_interp_len, plot_stride, plot_shap, sample_wise_norml, csfont, font, my_plot_ana


figure = 3
plot_save_path = './plot'

if figure == 0:
    '''
    plot all the data for 1024 segments
    '''
    path = './data/s_stride_data_nor.mat'
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(path, test_rate=0.2, val_rate=0.2, categ=0)
    acc_data = np.vstack([X_train, X_test, X_val])

    y_data = np.hstack([y_train, y_test, y_val])
    y_index = np.where(y_data == 0)[0]
    o_index = np.where(y_data == 1)[0]

    interp_acc = acc_data

    group_list = ['Adult', 'Older adult']
    index = 8  # 8 strides
    plot_stride(interp_acc[y_index, :, :], interp_acc[o_index, :, :], group_list, index)
    # plt.savefig(f'./{plot_save_path}/6stride_all.pdf')
    plt.show()

if figure == 1:
    '''
    plot all the data for 128 segments
    '''
    path = './data/o_stride_data_nor.mat'
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(path, test_rate=0.2, val_rate=0.2, categ=0)
    acc_data = np.vstack([X_train, X_test, X_val])

    y_data = np.hstack([y_train, y_test, y_val])
    y_index = np.where(y_data == 0)[0]
    o_index = np.where(y_data == 1)[0]

    interp_acc = acc_data
    group_list = ['Adult', 'Older adult']
    plot_stride(interp_acc[y_index, :, :], interp_acc[o_index, :, :], group_list, index =1)
    plt.savefig(f'./{plot_save_path}/1stride_all.pdf')
    plt.show()
    print()

if figure == 2:
    # one stride ana for cnn
    path = './result/cnn.npy'
    interp_ana = np.load(path)

    data_path = './data/o_stride_data_nor.mat'
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(data_path, 0.2, 0.2)

    window_size = 128
    my_plot_ana(interp_ana, X_test, window_size)
    plt.savefig('./plot/one_stride_anan.pdf')
    plt.show()

if figure == 3:
    # eight strides ana for cnn
    path = './result/gru.npy'
    interp_ana = np.load(path)

    # interp_ana = interp_ana[0]
    data_path = './data/e_stride_data_nor.mat'
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(data_path, 0.2, 0.2)
    window_size = 1024

    my_plot_ana(interp_ana, X_test, window_size)
    plt.savefig('./plot/8stride_anan.pdf')
    plt.show()

