"""File Handling used to save model scores from parameter fitting values to csv files for data storage

@author: Zachary Spalding
"""

import csv
from operator import itemgetter
import numpy as np

class ParamCSVFile():
    
    def __init__(self, filename, newFile):
        self.filename = filename
        self.newline = ''
        self.newFile = newFile

    def readFromFile(self, paramGrouping=0):
        names = []
        vals = []
        
        # open csv file
        with open(self.filename, 'r', newline=self.newline) as f:
            # read in data rows as a dictionary
            reader = csv.reader(f)

            #save header data and row data (each row of vals is a different parameter set)
            noHeader = True
            for row in reader:
                if noHeader:
                    names = list(row)
                    noHeader = False
                else:
                    vals.append(list(row))
            vals = np.array(vals)

        #Group parameters by type rather than keeping parameter sets together
        if paramGrouping == 1:
            # create dictionary from 2D list (each name corresponds to list
            # of same value type rather than a parameter set)
            paramDict = {}
            for i in range(len(names)):
                paramDict[names[i]] = list(vals[:,i])

            return paramDict    
        else: 
            return vals    

    def writeHeader(self):
        paramNameList = ['R_max', 'Roff_max', 'W_max', 'S_max', 'tau_Roff', 'tau_R', 
                    'tau_W', 'tau_S', 'alpha_Roff', 'alpha_R', 'alpha_W', 'beta_R',
                    'beta_W', 'alpha_S', 'gamma_R', 'gamma_Roff', 'gamma_W', 'gamma_S',
                    'k1_Roff', 'k2_Roff', 'k1_S', 'k2_S', 'stp_max', 'stp_min', 'stp_r',
                    'tau_stpW', 'h_max', 'h_min', 'omega_max', 'omega_min', 'theta_R',
                    'theta_W', 'tau_stpup', 'tau_stpdown', 'tau_hup', 'tau_hdown', 'tau_omega',
                    'tau_stim', 'g_Roff2R', 'g_R2Roff', 'g_S2W', 'g_W2S', 'g_W2R', 'g_R2W', 
                    'g_W2Roff', 'g_Roff2W', 'g_Roff2S', 'tau_CR', 'tau_CRf', 'tau_CRoff',
                    'tau_CW', 'tau_CS', 'delta_update', 'Score']

        with open(self.filename, 'w', newline=self.newline) as f:
            writer = csv.writer(f)
            writer.writerow(paramNameList)       

    def writeToFile(self, model, score): 
        dataDict = model.paramDict
        dataDict['Score'] = score

        with open(self.filename, 'a', newline=self.newline) as f:
            writer = csv.writer(f)
            if self.newFile:
                self.writeHeader()
                self.newFile = False
            writer.writerow(list(dataDict.values()))

        print(f'Data Written to {self.filename}! \n')

    def getParamAverages(self):
        groupedParams = self.readFromFile(paramGrouping=1)

        avgs = []
        for _, value in groupedParams.items():
            avgs.append(np.average(value))

        avgDict = {}
        names = list(groupedParams.keys())
        for i in range(len(names)):
            avgDict[names[i]] = avgs[i]

        return avgDict

    def getMinParamSet(self):
        paramSets = self.readFromFile()
        minScoreInd = np.argmin(paramSets[:, -1])
        return paramSets[minScoreInd, :]

    def clearFile(self):
        with open(self.filename, 'w', newline=self.newline) as f:
            f.truncate()