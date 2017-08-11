
from salamanca.models.calibrate_ineq import Model, Model1


def data():
    natdata = pd.Series({
        'n': 15,
        'i': 105,
        'gini': 0.4,
    })
    subdata = pd.DataFrame({
        'n': [5, 10],
        'i': [10, 5],
        'gini': [0.3, 0.5],
    })
    return natdata, subdata


def test_model_data():
    natdata, subdata = data()
    model = Model(natdata, subdata)
