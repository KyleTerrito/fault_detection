from ctypes import alignment
from re import I
from cmd2 import style
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib import cm
from matplotlib.pyplot import gca
from matplotlib.spines import Spine
from matplotlib.ticker import (AutoLocator, AutoMinorLocator,
                               FormatStrFormatter, MultipleLocator)
import matplotlib.ticker
from typing import Optional
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition, inset_axes,
                                                   mark_inset)
from math import log

import pickle


class Plotters():
    '''
    Utilities class to handle all plots.
    '''
    def __init__(self):
        pass

    def plot_metrics_opt(self, metric):
        fig = plt.figure(figsize=(6, 5))
        axs0 = fig.add_subplot(1, 1, 1)

        metrics_dict = {'sil_score': 0, 'ch_score': 1, 'dbi_score': 2}
        y_label_dict = {
            'sil_score': r'$\rm Silhouette\ Score$',
            'ch_score': r'$\rm  Normalized\ Calinski-Harabasz\ Index$',
            'dbi_score': r'$\rm Normalized\ Davies-Bouldin\ Index$'
        }

        dr_methods = ['NO DR', 'PCA', 'UMAP']
        cl_methods = ['KMEANS', 'DBSCAN', 'HDBSCAN']

        for dr_method in dr_methods:
            for cl_method in cl_methods:
                metrics = pickle.load(
                    open(
                        f'tests/ensemble_test_results/metrics{dr_method}_{cl_method}.pkl',
                        'rb'))

                metrics_sorted = metrics.transpose().sort_values(
                    by=[f'{dr_method}_{cl_method}_acc']).to_numpy()

                if metric is not 'sil_score':

                    axs0.scatter(-1 * metrics_sorted[:, -1],
                                 metrics_sorted[:, metrics_dict[metric]] /
                                 max(metrics_sorted[:, metrics_dict[metric]]),
                                 marker='s',
                                 label=f'{dr_method}-{cl_method}',
                                 edgecolor='k',
                                 linewidths=0.5)

                    axs0.set_ylim(0, 1)
                else:
                    axs0.scatter(-1 * metrics_sorted[:, -1],
                                 metrics_sorted[:, metrics_dict[metric]],
                                 marker='s',
                                 label=f'{dr_method}-{cl_method}',
                                 edgecolor='k',
                                 linewidths=0.5)

        axs0.set_ylabel(y_label_dict[metric],
                        labelpad=5,
                        fontsize='x-large',
                        fontname='Times New Roman')

        axs0.set_xlabel(r'$\rm Accuracy$',
                        labelpad=5,
                        fontsize='x-large',
                        fontname='Times New Roman')

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        axs0.set_xlim(0, )
        axs0.legend(
            frameon=False,
            loc='best',
            ncol=1,
            fontsize='x-small',
            #bbox_to_anchor=(0.5, 0.5)
        )
        # plt.tight_layout()
        # ax2.legend(frameon = False, loc=1, fontsize = 'medium')
        #plt.legend(frameon = False)
        plt.savefig(f'reports/{metric}.pdf', bbox_inches='tight')
        plt.savefig(f'reports/{metric}.png',
                    transparent=True,
                    bbox_inches='tight')

        plt.show()

    def plot_metrics(self, res_dict, reconstruction_errors, sil_scores,
                     CH_scores, DBI_scores, n_clusters):

        key_names_all = [
            'NO DR KMEANS',
            'NO DR DBSCAN',
            'NO DR HDBSCAN',
            'PCA KMEANS',
            'PCA DBSCAN',
            'PCA HDBSCAN',
            'UMAP KMEANS',
            'UMAP DBSCAN',
            'UMAP HDBSCAN',
        ]

        key_names_format = [
            'NO DR\n KMEANS',
            'NO DR\n DBSCAN',
            'NO DR\n HDBSCAN',
            'PCA\n KMEANS',
            'PCA\n DBSCAN',
            'PCA\n HDBSCAN',
            'UMAP\n KMEANS',
            'UMAP\n DBSCAN',
            'UMAP\n HDBSCAN',
        ]

        k_dict = {
            z[0]: list(z[1:])
            for z in zip(key_names_all, key_names_format)
        }

        if reconstruction_errors is not None:
            figname = 'Fig3'
            fig = plt.figure(figsize=(10, 5))
            axs0 = fig.add_subplot(1, 2, 1)
            axs1 = fig.add_subplot(1, 2, 2)
            i = 0

            for key, value in res_dict.items():

                x = k_dict[' '.join(key)]

                if 'UMAP' in x[0]:
                    if reconstruction_errors[i] > 0:
                        axs1.bar(x,
                                 reconstruction_errors[i],
                                 align='center',
                                 width=0.3)

                    for tick in axs1.get_xticklabels():
                        tick.set_fontname("Times New Roman")
                    for tick in axs1.get_yticklabels():
                        tick.set_fontname("Times New Roman")

                    axs1.yaxis.set_major_locator(AutoLocator())
                    axs1.yaxis.set_minor_locator(AutoMinorLocator())
                    axs1.tick_params(direction='out',
                                     pad=10,
                                     length=9,
                                     width=1.0,
                                     labelsize='large')
                    axs1.tick_params(which='minor',
                                     direction='out',
                                     pad=10,
                                     length=5,
                                     width=1.0,
                                     labelsize='large')

                    for axis in ['top', 'bottom', 'left', 'right']:
                        axs1.spines[axis].set_linewidth(1.0)

                else:
                    if reconstruction_errors[i] > 0:
                        axs0.bar(x,
                                 reconstruction_errors[i],
                                 align='center',
                                 width=0.3)

                i += 1

            axs0.set_ylabel(r'$\rm Reconstruction\ error\ (MSE)$',
                            labelpad=5,
                            fontsize='x-large',
                            fontname='Times New Roman')

            axs1.set_ylabel(r'$\rm Reconstruction\ error\ (MSE)$',
                            labelpad=5,
                            fontsize='x-large',
                            fontname='Times New Roman')

            axs1.set_xlabel(r'$\rm Ensemble$',
                            labelpad=5,
                            fontsize='x-large',
                            fontname='Times New Roman')

            axs0.set_ylim(0, )
            axs1.set_ylim(0, )

        if sil_scores is not None:
            figname = 'Fig5'
            fig = plt.figure(figsize=(12, 5))
            axs0 = fig.add_subplot(1, 1, 1)
            i = 0

            for key, value in res_dict.items():
                x = k_dict[' '.join(key)]

                axs0.bar(x, sil_scores[i], align='center', width=0.3)
                i += 1

            axs0.set_ylabel(r'$\rm Silhouette\ score$',
                            labelpad=5,
                            fontsize='x-large',
                            fontname='Times New Roman')

            axs0.plot([-1, 9], [0, 0], color='k', linewidth=1.0)
            axs0.set_xlim(-1, 9)

        if CH_scores is not None:
            figname = 'Fig6'
            fig = plt.figure(figsize=(12, 5))
            axs0 = fig.add_subplot(1, 1, 1)
            i = 0

            for key, value in res_dict.items():
                x = k_dict[' '.join(key)]

                axs0.bar(x, CH_scores[i], align='center', width=0.3)
                i += 1

            axs0.set_ylabel(r'$\rm Calinski-Harabasz\ score$',
                            labelpad=5,
                            fontsize='x-large',
                            fontname='Times New Roman')

            axs0.plot([-1, 9], [0, 0], color='k', linewidth=1.0)
            axs0.set_xlim(-1, 9)

        if DBI_scores is not None:
            figname = 'Fig7'
            fig = plt.figure(figsize=(12, 5))
            axs0 = fig.add_subplot(1, 1, 1)
            i = 0

            for key, value in res_dict.items():
                x = k_dict[' '.join(key)]

                axs0.bar(x, DBI_scores[i], align='center', width=0.3)
                i += 1

            axs0.set_ylabel(r'$\rm Davies-Bouldin\ score$',
                            labelpad=5,
                            fontsize='x-large',
                            fontname='Times New Roman')

            axs0.plot([-1, 9], [0, 0], color='k', linewidth=1.0)
            axs0.set_xlim(-1, 9)

        if n_clusters is not None:
            figname = 'Fig4'
            fig = plt.figure(figsize=(12, 5))
            axs0 = fig.add_subplot(1, 1, 1)
            i = 0

            for key, value in res_dict.items():
                x = k_dict[' '.join(key)]

                axs0.bar(x, n_clusters[i], align='center', width=0.3)
                i += 1

            axs0.set_ylabel(r'$\rm Number\ of\ clusters$',
                            labelpad=5,
                            fontsize='x-large',
                            fontname='Times New Roman')

            axs0.plot([-1, 9], [0, 0], color='k', linewidth=1.0)
            axs0.set_xlim(-1, 9)

        axs0.set_xlabel(r'$\rm Ensemble$',
                        labelpad=5,
                        fontsize='x-large',
                        fontname='Times New Roman')

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        # Additional formatting
        # axs2.set_autoscale_on

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        # axs0.legend(
        #     frameon=False,
        #     loc='upper left',
        #     ncol=2,
        #     fontsize='x-small',
        #     #bbox_to_anchor=(0.5, 0.5)
        # )
        plt.tight_layout()
        # ax2.legend(frameon = False, loc=1, fontsize = 'medium')
        #plt.legend(frameon = False)
        plt.savefig(f'reports/{figname}.pdf', bbox_inches='tight')
        plt.savefig(f'reports/{figname}.png',
                    transparent=True,
                    bbox_inches='tight')

        plt.show()

        return None

    def plot_performance(self, res_dict, hyper_names):

        h_names_all = [
            'n_comp', 'n_neighbors', 'min_dist', 'n_components', 'n_clusters',
            'eps', 'min_samples', 'num_neighbors', 'min_cluster_size',
            'cluster_selection_epsilon'
        ]
        h_names_format = [
            '$n_{comp}$', '$n_{neighbors}^{UMAP}$', '$min_{dist}$',
            '$n_{components}$', '$n_{clusters}$', '$eps$', '$min_{samples}$',
            '$n_{neighbors}^{kNN}$', '$min_{cluster\ size}$', '$\u03B5$'
        ]

        h_dict = {z[0]: list(z[1:]) for z in zip(h_names_all, h_names_format)}

        key_names_all = [
            'NO DR KMEANS',
            'NO DR DBSCAN',
            'NO DR HDBSCAN',
            'PCA KMEANS',
            'PCA DBSCAN',
            'PCA HDBSCAN',
            'UMAP KMEANS',
            'UMAP DBSCAN',
            'UMAP HDBSCAN',
        ]

        key_names_format = [
            'NO DR\n KMEANS',
            'NO DR\n DBSCAN',
            'NO DR\n HDBSCAN',
            'PCA\n KMEANS',
            'PCA\n DBSCAN',
            'PCA\n HDBSCAN',
            'UMAP\n KMEANS',
            'UMAP\n DBSCAN',
            'UMAP\n HDBSCAN',
        ]

        k_dict = {
            z[0]: list(z[1:])
            for z in zip(key_names_all, key_names_format)
        }

        fig = plt.figure(figsize=(12, 5))

        axs0 = fig.add_subplot(1, 1, 1)

        i = 0

        for key, value in res_dict.items():

            hyper_names[i] = [h_dict[el] for el in hyper_names[i]]

            h_names = hyper_names[i]
            h_values = value[0]

            for k in range(len(h_values)):
                if isinstance(h_values[k], float):
                    h_values[k] = "{:.2f}".format(h_values[k])

            label = ' '.join([
                f'{h_names[j][0]} = {h_values[j]}, '
                for j in range(len(h_values))
            ])[:-2]

            accuracy = value[1]  #[el for el in value[1]]

            x = k_dict[' '.join(key)]

            axs0.bar(x, accuracy, align='center', width=0.3, label=label)
            i += 1

        axs0.set_ylabel(r'$\rm Accuracy$',
                        labelpad=5,
                        fontsize='x-large',
                        fontname='Times New Roman')

        axs0.set_xlabel(r'$\rm Ensemble$',
                        labelpad=5,
                        fontsize='x-large',
                        fontname='Times New Roman')

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        axs0.yaxis.set_major_locator(AutoLocator())
        axs0.yaxis.set_minor_locator(AutoMinorLocator())
        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        # Additional formatting
        # axs2.set_autoscale_on
        axs0.set_ylim(0, 1.8)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        axs0.legend(
            frameon=False,
            loc='upper left',
            ncol=2,
            fontsize='x-small',
            #bbox_to_anchor=(0.5, 0.5)
        )
        plt.tight_layout()
        # ax2.legend(frameon = False, loc=1, fontsize = 'medium')
        #plt.legend(frameon = False)
        figname = 'Fig2'
        plt.savefig(f'reports/{figname}.pdf', bbox_inches='tight')
        plt.savefig(f'reports/{figname}.png',
                    transparent=True,
                    bbox_inches='tight')

        plt.show()

        return None

    def restore_minor_ticks_log_plot(self,
                                     ax: Optional[plt.Axes] = None,
                                     n_subticks=9) -> None:
        """For axes with a logrithmic scale where the span (max-min) exceeds
        10 orders of magnitude, matplotlib will not set logarithmic minor ticks.
        If you don't like this, call this function to restore minor ticks.

        Args:
            ax:
            n_subticks: Number of Should be either 4 or 9.

        Returns:
            None
        """
        if ax is None:
            ax = plt.gca()

        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=1000)
        ax.xaxis.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(base=10.0,
                                              subs=np.linspace(
                                                  0, 1.0,
                                                  n_subticks + 2)[1:-1],
                                              numticks=1000)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    def plot_pareto(self, res_dict, hyper_names):

        h_names_all = [
            'n_comp', 'n_neighbors', 'min_dist', 'n_components', 'n_clusters',
            'eps', 'min_samples', 'num_neighbors', 'min_cluster_size',
            'cluster_selection_epsilon'
        ]
        h_names_format = [
            '$n_{comp}$', '$n_{neighbors}^{UMAP}$', '$min_{dist}$',
            '$n_{components}$', '$n_{clusters}$', '$eps$', '$min_{samples}$',
            '$n_{neighbors}^{kNN}$', '$min_{cluster\ size}$', '$\u03B5$'
        ]

        h_dict = {z[0]: list(z[1:]) for z in zip(h_names_all, h_names_format)}

        key_names_all = [
            'NO DR KMEANS',
            'NO DR DBSCAN',
            'NO DR HDBSCAN',
            'PCA KMEANS',
            'PCA DBSCAN',
            'PCA HDBSCAN',
            'UMAP KMEANS',
            'UMAP DBSCAN',
            'UMAP HDBSCAN',
        ]

        key_names_format = [
            'NO DR - KMEANS',
            'NO DR - DBSCAN',
            'NO DR - HDBSCAN',
            'PCA - KMEANS',
            'PCA - DBSCAN',
            'PCA - HDBSCAN',
            'UMAP - KMEANS',
            'UMAP - DBSCAN',
            'UMAP - HDBSCAN',
        ]

        colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive'
        ]

        colors_dict = dict(zip(key_names_format, colors))

        k_dict = {
            z[0]: list(z[1:])
            for z in zip(key_names_all, key_names_format)
        }

        fig = plt.figure(figsize=(12.5, 8.5))
        axs0 = fig.add_subplot(1, 1, 1)

        i = 0
        x_pareto_list = []
        y_pareto_list = []
        c_list = []

        labels_list = []
        methods_list = []
        legend_list = []

        for key, value in res_dict.items():

            hyper_names[i] = [h_dict[el] for el in hyper_names[i]]

            for val in value[0]:
                h_names = hyper_names[i]
                h_values = [el for el in val]

                for k in range(len(h_values)):
                    if isinstance(h_values[k], float):
                        h_values[k] = "{:.2f}".format(h_values[k])

                method = k_dict[' '.join(key)]
                methods_list.append(method)
                label = f'{method[0]}: ' + ' '.join([
                    f'{h_names[j][0]} = {h_values[j]}, '
                    for j in range(len(h_values))
                ])[:-2]

                labels_list.append(label)
                c_list.append(colors_dict[method[0]])
                legend_list.append(method[0][0])

            accuracies = (np.asarray(value[1])[:, 0])
            n_clusters = (np.asarray(value[1])[:, 1])

            x_pareto_list.extend(abs(accuracies))
            y_pareto_list.extend(abs(n_clusters))

            axs0.scatter(accuracies,
                         n_clusters,
                         s=100,
                         marker='s',
                         c=colors_dict[method[0]],
                         edgecolor='k',
                         linewidths=0.5,
                         alpha=0.9,
                         label=f'{method[0]}')

            i += 1

        axs0.scatter(
            x_pareto_list,
            y_pareto_list,
            s=100,
            marker='s',
            edgecolor='k',
            linewidths=0.5,
            c=c_list,
            alpha=0.9,
        )

        m = 0
        met_list = []

        met_list1 = [''.join([i for i in j[0:4]]) for j in labels_list]
        met_list2 = [''.join([i for i in j[4:14]]) for j in labels_list]
        x_pos = np.linspace(0.95, 1., len(met_list1))
        y_pos = np.linspace(-0.95, 1.1, len(met_list1))
        x_pos_dict = dict(zip(met_list1, x_pos))
        y_pos_dict = dict(zip(met_list2, y_pos))

        for i, txt in enumerate(labels_list):

            met = [i for i in txt[:14]]

            if x_pareto_list[i] > 0.99 and met not in met_list:

                axs0.annotate(
                    i, (x_pos_dict[''.join(met)[0:4]],
                        y_pareto_list[i] + y_pos_dict[''.join(met)[4:14]]),
                    color=c_list[i],
                    fontfamily='serif',
                    fontweight='bold')

                axs0.annotate(f'{i}: ' + txt, (170, 180 - m),
                              xycoords='figure points',
                              color=c_list[i],
                              fontfamily='serif',
                              fontweight='bold',
                              fontsize='small')

                met_list.append(met)

                m += 15

        axs0.set_yscale('log')
        axs0.set_ylabel(r'$\rm Number\ of\ clusters$',
                        labelpad=5,
                        fontsize='x-large',
                        fontname='Times New Roman')

        axs0.set_xlabel(r'$\rm Accuracy$',
                        labelpad=5,
                        fontsize='x-large',
                        fontname='Times New Roman')

        for tick in axs0.get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs0.get_yticklabels():
            tick.set_fontname("Times New Roman")

        self.restore_minor_ticks_log_plot(axs0)

        axs0.xaxis.set_major_locator(AutoLocator())
        axs0.xaxis.set_minor_locator(AutoMinorLocator())

        axs0.tick_params(direction='out',
                         pad=10,
                         length=9,
                         width=1.0,
                         labelsize='x-large')
        axs0.tick_params(which='minor',
                         direction='out',
                         pad=10,
                         length=5,
                         width=1.0,
                         labelsize='x-large')

        for axis in ['top', 'bottom', 'left', 'right']:
            axs0.spines[axis].set_linewidth(1.0)

        axs0.set_xlim(0, )
        axs0.set_ylim(0, )

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        axs0.legend(frameon=False,
                    loc='upper left',
                    ncol=2,
                    fontsize='x-small')
        #plt.tight_layout()

        figname = 'Fig2'
        plt.savefig(f'reports/{figname}.pdf', bbox_inches='tight')
        plt.savefig(f'reports/{figname}.png',
                    transparent=True,
                    bbox_inches='tight')

        plt.show()

        return None