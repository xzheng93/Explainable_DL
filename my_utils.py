import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn import metrics
import matplotlib
from scipy import interpolate
import matplotlib.colors as colors
from sklearn import preprocessing
from matplotlib.collections import LineCollection
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import matplotlib.font_manager as font_manager

random_state = 0
csfont = {'fontname': 'Times New Roman'}
font = font_manager.FontProperties(family='Times New Roman', size=16)


# plots confusion matrix
def plot_cm(cm, label, title):
    fig, ax = plt.subplots()
    im, cbar, cf_percentages, accuracy = heatmap(cm, label, label, title=title,
                                                 ax=ax,
                                                 cmap="Blues", cbarlabel="percentage [%]")
    annotate_heatmap(im, cf_percentages, cm, valfmt="{x:.1f} t")
    fig.tight_layout()


def heatmap(data, row_labels, col_labels, title, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    cf_percentages = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cf_percentages * 100, **kwargs)
    im.set_clim(0, 100)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    ax.set_xticklabels(col_labels, fontsize='small')
    ax.set_yticklabels(row_labels, fontsize='small')

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.ylabel('True label', fontsize='large', **csfont)
    plt.xlabel('Predicted label', fontsize='large', **csfont)
    accuracy = np.trace(data) / float(np.sum(data))
    stats_text = "\nAccuracy={:0.1%}".format(accuracy)
    plt.title(title + stats_text, **csfont)

    return im, cbar, cf_percentages, accuracy


def annotate_heatmap(im, cf_percentages, data=None, valfmt="{x:.1f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(cf_percentages)
    else:
        threshold = cf_percentages.max() / 2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Change the text's color depending on the data.
    cf_percentages = data.astype('float') / data.sum(axis=1)[:, np.newaxis]
    group_percentages = ["{0:.1%} \n".format(value) for value in cf_percentages.flatten()]
    group_counts = ["{0:0.0f}\n".format(value) for value in data.flatten()]
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_percentages, group_counts)]
    box_labels = np.asarray(box_labels).reshape(data.shape[0], data.shape[1])

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(cf_percentages[i, j] > threshold)])
            text = im.axes.text(j, i, box_labels[i, j], **kw)
            texts.append(text)

    return texts


# plot roc curve for one classification results
def plot_roc(fpr, tpr, title):
    plt.figure(figsize=(7, 6))
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, c='r', lw=3, alpha=0.7, label=u'AUC=%.2f' % auc)
    plt.plot((0, 1), (0, 1), c='b', lw=2, ls='--', alpha=0.7, label='baseline = 0.5')
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['right'].set_linewidth(1.3)
    ax.spines['top'].set_linewidth(1.3)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.grid()
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
    plt.title(title, fontsize=16)
    return auc


def my_sample(subjects, my_list):
    data_index = []
    for i in range(0, len(my_list)):
        m_index, = np.where(subjects == my_list[i])
        data_index.extend(m_index)
    return np.array(data_index)


def load_data(path, test_rate=0.2, val_rate=0.1, categ=1):
    data = scio.loadmat(path)
    stride_data = data['acc']
    age = data['age'].flatten()
    sub_id = data['id'].flatten()

    groups = np.zeros(len(age))
    groups[age > 65] = 1
    if categ == 1:
        groups = to_categorical(groups, 2)
    uid, uid_index = np.unique(sub_id, return_index=True)

    # split the subjects id to train, validation and test
    trVa_id, test_id, trVa_id_y, test_id_y = train_test_split(uid, groups[uid_index],
                                                              random_state=random_state,
                                                              stratify=groups[uid_index],
                                                              test_size=test_rate)

    train_id, val_id, train_id_y, val_id_y = train_test_split(trVa_id, trVa_id_y,
                                                              random_state=random_state,
                                                              stratify=trVa_id_y,
                                                              test_size=val_rate / (1 - test_rate))

    train_index = my_sample(sub_id, train_id)
    val_index = my_sample(sub_id, val_id)
    test_index = my_sample(sub_id, test_id)

    shuffle(train_index)
    shuffle(test_index)
    shuffle(val_index)

    X_train = stride_data[train_index]
    X_test = stride_data[test_index]
    X_val = stride_data[val_index]

    y_train = groups[train_index]
    y_test = groups[test_index]
    y_val = groups[val_index]

    return X_train, X_test, X_val, y_train, y_test, y_val


def my_interp(data, index_list, ip_size):
    interp_data = []
    for i in range(0, len(data)):
        x = interpolate.UnivariateSpline(range(0, index_list[i]), data[i, 0:index_list[i], 0], s=0)
        y = interpolate.UnivariateSpline(range(0, index_list[i]), data[i, 0:index_list[i], 1], s=0)
        z = interpolate.UnivariateSpline(range(0, index_list[i]), data[i, 0:index_list[i], 2], s=0)
        x = x(np.linspace(0, index_list[i] - 1, ip_size))
        y = y(np.linspace(0, index_list[i] - 1, ip_size))
        z = z(np.linspace(0, index_list[i] - 1, ip_size))
        interp_data.append(np.array([x, y, z]).T)
    interp_data = np.array(interp_data)
    return interp_data


def plot_stride(acc_data1, acc_data2, group_list, index):
    for i in range(0, len(acc_data1)):
        acc_data1[i] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(acc_data1[i])
    for i in range(0, len(acc_data2)):
        acc_data2[i] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(acc_data2[i])

    leg_list = ['Vertical acc (normalized)', 'ML acc (normalized)', 'AP acc (normalized)']
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    x_tick = np.arange(0, acc_data1.shape[1]) / acc_data1.shape[1] * 100*index

    for i in range(0, 2):
        if i == 0:
            data = acc_data1
        else:
            data = acc_data2
        mean_acc = np.mean(data, axis=0)
        std_acc = np.std(data, axis=0)

        for j in range(0, 3):
            # axs[i, j].set_ylim([np.min(mean_acc[:, j]) - 0.5, np.max(mean_acc[:, j]) + 0.5])
            axs[i, j].plot(x_tick, mean_acc[:, j], 'b', label=f'{group_list[i]} mean', zorder=3)
            axs[i, j].fill_between(x_tick, mean_acc[:, j] - std_acc[:, j], mean_acc[:, j] + std_acc[:, j],
                                   zorder=2, label='std', color='mistyrose')
            # for k in range(0, int(len(data)/10)):
            #     axs[i, j].plot(data[k*9, :, j], color='whitesmoke', zorder=1, linewidth=0.1)

            axs[i, j].legend(fontsize=13, prop=font)
            axs[i, j].set_xlabel('Gait cycle [%]', fontsize=18, **csfont)
            axs[i, j].set_ylabel(leg_list[j], fontsize=18, **csfont)
            # axs[i, j].set_ylim([-0.2, 1.2])
            axs[i, j].set_xlim([0, 100*index])
    fig.tight_layout()
#
#
# def plot_o_stride_ana(acc_data1, acc_data2, ana_data1, ana_data2, group_list):
#     for i in range(0, len(acc_data1)):
#         acc_data1[i] = MinMaxScaler().fit_transform(acc_data1[i])
#     for i in range(0, len(acc_data2)):
#         acc_data2[i] = MinMaxScaler().fit_transform(acc_data2[i])
#
#     leg_list = ['Vertical acc (normalized)', 'ML acc (normalized)', 'AP acc (normalized)']
#     fig, axs = plt.subplots(3, 3, figsize=(16, 13))
#     x_tick = np.arange(0, acc_data1.shape[1]) / acc_data1.shape[1] * 100
#     # cmap = colors.LinearSegmentedColormap.from_list('my_color', [(0, '#0404FF'), (0.35, '#F0F0FF'), (0.5, '#dedfe0'), (0.65, '#FFECEC'), (1, '#FF0404')], N=256)
#
#     for i in range(0, 3):
#         if i == 0:
#             mean_acc1 = np.mean(acc_data1, axis=0)
#             mean_acc2 = np.mean(acc_data2, axis=0)
#             for j in range(0, 3):
#                 axs[i, j].plot(x_tick, mean_acc1[:, j], 'r', label=f'{group_list[0]} acc mean')
#                 axs[i, j].plot(x_tick, mean_acc2[:, j], 'b', label=f'{group_list[1]} acc mean')
#                 axs[i, j].set_ylabel(leg_list[j], fontsize=18)
#                 axs[i, j].set_xlabel('Gait cycle [%]', fontsize=18)
#                 axs[i, j].set_ylabel(leg_list[j], fontsize=18)
#                 axs[i, j].legend(fontsize=13)
#
#         elif 1 <= i <= 2:
#             if i == 1:
#                 ana_data = ana_data1
#                 acc_data = acc_data1
#             elif i == 2:
#                 ana_data = ana_data2
#                 acc_data = acc_data2
#             mean_anamean_ana_all = np.mean(ana_data, axis=0)
#             mean_anamean_ana_all = mean_anamean_ana_all[None, :, :]
#             mean_ana_all = sample_wise_norml(mean_anamean_ana_all)[0, :, :]
#
#             for j in range(0, 3):
#                 acc = acc_data[:, :, j]
#                 mean_ana = mean_ana_all[:, j]
#                 mean_acc = np.mean(acc, axis=0)
#                 # mean_ana = preprocessing.MaxAbsScaler().fit_transform(mean_ana.reshape(-1, 1)).flatten()
#                 points = np.array([x_tick, mean_acc]).T.reshape(-1, 1, 2)
#                 segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
#                 norm = plt.Normalize(-1, 1)
#                 lc = LineCollection(segments, cmap='bwr', norm=norm)
#                 lc.set_array(mean_ana)
#                 lc.set_linewidth(8)
#                 # lc.set_alpha(0.8)
#                 line = axs[i, j].add_collection(lc)
#
#                 axs[i, j].plot(x_tick, mean_acc, 'black', label=f'{group_list[i - 2]} acc mean', alpha=0.5)
#                 axs[i, j].set_ylim([0, 1])
#                 axs[i, j].legend(fontsize=13)
#                 axs[i, j].set_xlabel('Gait cycle [%]', fontsize=18)
#                 axs[i, j].set_ylabel(leg_list[j], fontsize=18)
#             fig.colorbar(line, ax=axs[i, j])
#
#     fig.tight_layout()
#
#
# def plot_o_stride_ana_bar(ana_data1, ana_data2):
#     ana_data = np.vstack([ana_data1, ana_data2])
#     ana_mean = np.mean(np.abs(ana_data), axis=0)
#     ana_mean = ana_mean[None, :, :]
#     ana_mean = sample_wise_norml(ana_mean)[0, :, :]
#     ana_sum = np.sum(ana_mean)
#     ana_per = ana_mean / ana_sum * 100
#     x_ticks = np.arange(len(ana_per))
#     # ana_per_bar = [np.sum(ana_per[int(len(ana_mean) * m*0.1): int(len(ana_mean)*(m + 1)*0.1), :], axis=0)
#     #                for m in range(0, 10)]
#     # ana_per_bar = np.array(ana_per_bar)
#     #
#     # x_ticks = (np.arange(10)+0.5)*10
#
#     width = 1
#     p1 = plt.bar(x_ticks, ana_per[:, 0], width, color='#3FAA59', alpha=0.8)
#     p2 = plt.bar(x_ticks, ana_per[:, 1], width, bottom=ana_per[:, 0], color='#F7A409', alpha=0.8)
#     p3 = plt.bar(x_ticks, ana_per[:, 2], width, bottom=ana_per[:, 1] + ana_per[:, 0], color='#E54E35', alpha=0.8)
#
#     plt.ylabel('abs(lrp) percentage [%]')
#     # plt.title('Scores by group and gender')
#     plt.xticks(np.array([0, 1, 2.5, 5, 6, 7.5, 10]) * 25.6,
#                ('Heel strike', 'Toes off', 'Mid stance', 'Heel strike', 'Toes off', 'Mid stance', 'Heel strike'),
#                rotation=70)
#     plt.xlim([0, 256])
#     # plt.yticks(np.arange(0, 81, 10))
#     plt.legend((p1[0], p2[0], p3[0]), ('V', 'ML', 'AP'))
#     plt.tight_layout()
#
#
def plot_shap(y_acc, o_acc, interp_ana, group_list, window_size, title=''):
    # normalize the data into 0-1
    for i in range(0, len(y_acc)):
        y_acc[i] = MinMaxScaler().fit_transform(y_acc[i])
    for i in range(0, len(o_acc)):
        o_acc[i] = MinMaxScaler().fit_transform(o_acc[i])
    cmap = colors.LinearSegmentedColormap.from_list( 'my_color',[(0, 'silver'),(0.25, 'silver'),(0.35, '#e4e631'),  (0.45, '#e99c06'), (0.75, '#fe1200'),(1, '#FF0404'),], N=256)

    leg_list = ['Mean_Acc_V (normalized)', 'Mean_Acc_ML (normalized)', 'Mean_Acc_AP (normalized)']
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    x_tick = np.arange(0, y_acc.shape[1]) / y_acc.shape[1] * 100

    mean_acc1 = np.mean(y_acc, axis=0)
    mean_acc2 = np.mean(o_acc, axis=0)

    mean_anamean_ana_all = np.mean(abs(interp_ana), axis=0)
    mean_anamean_ana_all = mean_anamean_ana_all[None, :, :]
    ana = sample_wise_norml(mean_anamean_ana_all)[0, :, :]

    for j in range(0, 3):
        axs[j].plot(x_tick, mean_acc1[:, j], 'r', label=f'{group_list[0]}')
        axs[j].plot(x_tick, mean_acc2[:, j], 'b', label=f'{group_list[1]}')
        axs[j].set_xlabel('One Stride [%]', fontsize=18, **csfont)
        axs[j].set_ylabel(leg_list[j], fontsize=18, **csfont)
        L=axs[j].legend(fontsize=13, loc='upper left')
        plt.setp(L.texts, **csfont)

        points = np.array([x_tick, 0.05*np.ones(window_size)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(ana[:, j])
        lc.set_linewidth(20)
        line = axs[j].add_collection(lc)
        axs[j].set_ylim([0, 1])
        axs[j].legend(fontsize=13, prop=font)
    plt.legend(fontsize=13, loc='upper left')
    fig.colorbar(line, ax=axs[j]).set_label(label='Mean abs(SHAP values)',size=15,**csfont)
    plt.title(title)
    fig.tight_layout()


def my_plot_ana(interp_ana, X_test, window_size):
    interp_ana = np.abs(interp_ana)
    interp_ana_m = np.mean(interp_ana, axis=0)
    mean_ana_all = interp_ana_m[None, :, :]
    ana = sample_wise_norml(mean_ana_all)[0, :, :]  # normalize across [xxxx, yyyy, zzzz]

    ana_3in1 = np.sum(interp_ana_m, axis=1)
    if window_size == 1024:
        ana_3in1 = [ana_3in1[n*128:(n+1)*128] for n in range(0,8)]
        ana_3in1 = np.array(ana_3in1)
        ana_3in1 = np.sum(ana_3in1,  axis=0)
    ana_3in1 = MinMaxScaler().fit_transform(ana_3in1.reshape(-1, 1)).flatten()

    # normalize testing data for plot
    for i in range(0, len(X_test)):
        X_test[i] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_test[i])
    mean_acc = np.mean(X_test, axis=0)
    # define gait events
    phase6 = [0, 10, 30, 50, 60, 80, 100]
    phase6_l = ['initial contact', 'opp toe off', 'heel rise', 'opp contact', 'toe off', 'opp heel rise',
                'next init contact']
    cmap = colors.LinearSegmentedColormap.from_list( 'my_color',[(0, 'silver'),(0.20, 'silver'),(0.30, '#e4e631'),  (0.45, '#e99c06'), (0.65, '#fe1200'),(1, '#FF0404'),], N=256)
    leg_list = ['Mean V', 'Mean ML', 'Mean AP', 'AMP']
    if window_size == 128:
        x_tick = np.arange(0, X_test.shape[1]) / (X_test.shape[1]-1) * 100

    if window_size == 1024:
        x_tick = np.arange(0, X_test.shape[1]) / (X_test.shape[1]-1) * 800

    points = np.array([x_tick, -0.95 * np.ones(window_size)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    norm = plt.Normalize(0, 1)
    for j in range(0, 3):
        axs[j].plot(x_tick, mean_acc[:, j], 'r')
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(ana[:, j])
        lc.set_linewidth(20)
        line = axs[j].add_collection(lc)
        axs[j].set_ylim([-1, 1])
        fig.colorbar(line, ax=axs[j]).set_label(label='Mean abs(SHAP)', size=15, **csfont)
        axs[j].set_ylabel(leg_list[j], fontsize=18,  **csfont)
        axs[j].set_xlim([0, window_size/128*100])

        for n in phase6:
            if n == 50:
                axs[3].axvline(x=n, color='r', linestyle=":")
            else:
                axs[3].axvline(x=n, color='b', linestyle=":")
    axs[2].set_xlabel('One Cycle [%]', fontsize=18, **csfont)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(ana_3in1)
    lc.set_linewidth(20)
    line = axs[3].add_collection(lc)
    axs[3].set_ylim([-1, 1])
    fig.colorbar(line, ax=axs[3]).set_label(label='Mean abs(SHAP)',size=15,**csfont)
    axs[3].set_xlim([0, 100])
    axs[3].axis('off')
    axs[3].set_ylabel(leg_list[3], fontsize=18,  **csfont)
    for n in phase6:
        if n == 50:
            axs[3].axvline(x=n, color='r', linestyle=":")
        else:
            axs[3].axvline(x=n, color='b', linestyle=":")

    fig.tight_layout()


def find_interp_len(vector):
    def find_not_zero(vec):
        end = len(vec) - 1
        for i in range(0, end + 1):
            if vec[end - i] != 0:
                return end - i

    index_list = []
    for i in range(0, vector.shape[1]):
        index_list.append(find_not_zero(vector[:, i]))

    result = index_list.count(index_list[0]) == len(index_list)
    if result:
        return index_list[0]
    else:
        return np.min(index_list)


# normalize the ana data by [x,y,z] sample, so the differences between
# different axes can be reserved
def sample_wise_norml(data):
    for i in range(0, len(data)):
        a = np.hstack([data[i, :, 0], data[i, :, 1], data[i, :, 2]])
        scale = preprocessing.MaxAbsScaler().fit(a.reshape(-1, 1))
        data[i, :, 0] = scale.transform(data[i, :, 0].reshape(-1, 1)).flatten()
        data[i, :, 1] = scale.transform(data[i, :, 1].reshape(-1, 1)).flatten()
        data[i, :, 2] = scale.transform(data[i, :, 2].reshape(-1, 1)).flatten()
    return data
