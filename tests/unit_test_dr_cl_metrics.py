from src.autoML import Metrics
from src.dataPreprocessing import DataPreprocessing
from src.plotters import Plotters
import pickle


def test():

    preprocessor = DataPreprocessing()

    #X_train, X_test, y_train, y_test = preprocessor.load_data()

    X_train = pickle.load(
        open('tests/ensemble_test_results/X_train_file.pkl', 'rb'))
    res_dict = pickle.load(
        open('tests/ensemble_test_results/res_dict.pkl', 'rb'))

    metrics = Metrics()

    rc_error, sil_scores, n_clusters = metrics.get_metrics(res_dict=res_dict,
                                                           X_train=X_train)

    #print(n_clusters)
    plotter = Plotters()

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=rc_error,
                         sil_scores=None,
                         n_clusters=None)

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=None,
                         sil_scores=sil_scores,
                         n_clusters=None)

    plotter.plot_metrics(res_dict=res_dict,
                         reconstruction_errors=None,
                         sil_scores=None,
                         n_clusters=n_clusters)

    return None
