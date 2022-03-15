from src.autoKNN import AutoKNN
import pickle


def test():
    knn = AutoKNN()
    knn.get_data(path='data/processed/TEP_Selected_Faults.xlsx')

    hyperparameters_list = pickle.load(
        open('tests/ensemble_test_results/h_list.pkl', 'rb'))

    res_dict = pickle.load(
        open('tests/ensemble_test_results/res_dict.pkl', 'rb'))

    knn.plot_pareto(res_dict, hyperparameters_list)

    knn.plot_results()

    return None
