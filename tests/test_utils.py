from salamanca import utils


def test_csv_backend_fname():
    db = utils.CSVBackend()
    obs = db.fname('foo', 'bar')
    exp = 'foo_bar.csv'
    assert obs == exp
