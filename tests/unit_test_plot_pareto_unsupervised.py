from autocluster.plotters import Plotters
import pickle


def test():

    exp = 'aiche_pyro_ul'
    plotter = Plotters(exp=exp)

    with open(f'tests/ensemble_test_results/{exp}_h_list.pkl', 'rb') as handle:
        hyperparameters_list = pickle.load(handle)

    with open(f'tests/ensemble_test_results/{exp}_res_dict.pkl', 'rb') as handle:
        res_dict = pickle.load(handle)

    plotter.plot_pareto_unsupervised(res_dict, hyperparameters_list)
    
    with open(f'tests/ensemble_test_results/{exp}_h_list.pkl', 'rb') as handle:
        hyperparameters_list = pickle.load(handle)

    with open(f'tests/ensemble_test_results/{exp}_res_dict.pkl', 'rb') as handle:
        res_dict = pickle.load(handle)

    plotter.plot_pareto_unsupervised(res_dict, hyperparameters_list)


    return None
