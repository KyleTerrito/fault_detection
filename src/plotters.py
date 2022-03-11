from ctypes import alignment
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
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition, inset_axes,
                                                   mark_inset)
from xarray import align


class Plotters():
    '''
    Utilities class to handle all plots.
    '''
    def __init__(self):
        pass

    def plot_metrics(self, res_dict, reconstruction_errors, sil_scores,
                     n_clusters):

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

        #Additional formatting
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

            accuracy = [el for el in value[1]]

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

        #Additional formatting
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
