import unittest
import sys
from unittests.LoggerTests import LoggerTest
from unittests.ModelTests import ModelTest
from unittests.ApiTests import ApiTest

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LoggerTest))
    suite.addTest(unittest.makeSuite(ModelTest))
    suite.addTest(unittest.makeSuite(ApiTest))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
