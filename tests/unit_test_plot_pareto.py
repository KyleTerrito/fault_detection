from src.plotters import Plotters
import pickle


def test():

    #knn.get_data(path='data/processed/TEP_Selected_Faults.xlsx')

    exp = 'default'

    plotter = Plotters(exp=exp)

    hyperparameters_list = pickle.load(
        open(f'tests/ensemble_test_results/{exp}_h_list.pkl', 'rb'))

    res_dict = pickle.load(
        open(f'tests/ensemble_test_results/{exp}_res_dict.pkl', 'rb'))

    plotter.plot_pareto(res_dict, hyperparameters_list)

    hyperparameters_list = pickle.load(
        open(f'tests/ensemble_test_results/{exp}_h_list.pkl', 'rb'))

    res_dict = pickle.load(
        open(f'tests/ensemble_test_results/{exp}_res_dict.pkl', 'rb'))

    plotter.plot_pareto(res_dict, hyperparameters_list)

    return None
