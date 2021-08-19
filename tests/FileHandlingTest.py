import unittest

import sys
sys.path.append(r'C:\Users\Zac\Documents\Penn\Weber Lab\Flip-Flop Model\Code')
from FileHandling import ParamCSVFile

class TestParamCSVFile(unittest.TestCase):

    def testInit(self):
        csv = ParamCSVFile('test.csv')
        self.assertEqual(csv.filename, 'test.csv')
        self.assertEqual(csv.newline, '')

    def testReadFromFile(self):
        csv = ParamCSVFile('test.csv')
        vals = csv.readFromFile()
        self.assertEqual(vals, [1, 2])

    def testReadFromFileGrouped(self):
        csv = ParamCSVFile('test.csv')
        vals = csv.readFromFile()
        self.assertEqual(names, ['a', 'b'])
        self.assertEqual(vals, [1, 2])

    def testWriteToFile(self):
        csv = ParamCSVFile('testWrite1.csv')
        csvDict = {'a': 2, 'b': 3, 'c': 4}
        csv.writeToFile(csvDict)
        valDict = csv.readFromFile()
        self.assertEqual(valDict, {'a': [2], 'b': [3], 'c': [4]})

    def testWriteToFileMultiple(self):
        csv = ParamCSVFile('testWrite2.csv')
        csvDict1 = {'a': 2, 'b': 3, 'c': 4}
        csv.writeToFile(csvDict1)
        csvDict2 = {'a': 5, 'b': 6, 'c': 7}
        valDict = csv.readFromFile()
        self.assertEqual(valDict, {'a': [2,5], 'b': [3,6], 'c': [4,7]})

    def testGetParamAverages(self):
        csv = ParamCSVFile('testParam.csv')
        avgs = csv.getParamAverages()
        self.assertEqual(avgs, [2, 2, 2])

    if __name__ == '__main__':
        unittest.main()