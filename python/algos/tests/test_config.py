from algos.rnn.config import load_config

def test_load_config():
    config = load_config()

    assert config is not None
