from copyreg import pickle
from autocluster.plotters import Plotters
import pickle


def test():
    hyperparameters_list = pickle.load(
        open('tests/ensemble_test_results/h_list.pkl', 'rb'))
    res_dict = pickle.load(
        open('tests/ensemble_test_results/res_dict.pkl', 'rb'))
    '''
    Start - Plot results---------------------------------------------
    '''
    plotter = Plotters()

    plotter.plot_performance(res_dict, hyperparameters_list)
    '''
    End - Plot results---------------------------------------------
    '''
    return None