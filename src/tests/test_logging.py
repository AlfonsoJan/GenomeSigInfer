
from DeepBayesMutSig.logging import SingletonLogger

def test_singleton_instance():
    logger1 = SingletonLogger()
    logger2 = SingletonLogger()
    assert logger1 is logger2, "Instances are not the same"
