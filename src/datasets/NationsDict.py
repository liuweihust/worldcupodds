import csv
import sys
import json
import pandas as pd
import numpy as np

nationfile='nations.csv'

matchfilekey=['Host','Guest']
matchfiles=['worldcup_group.csv','worldcup_final.csv','worldcup_pred.csv']

groupfilekey=['M1','M2','M3','M4']
groupfile='group.csv'
csvfile='nations.csv'

class nations():
    def __init__(self,data_dir='../data/'):
        self._data=None
        self._listdata=[]
        self._datadir=data_dir
       
    def loadMatchFiles(self):
        for mfile in matchfiles:
            data = pd.read_csv(self._datadir+mfile, header=0)
            for i in range(data.shape[0]):
                for key in matchfilekey:
                    if data[key][i] not in self._listdata:
                        self._listdata.append(data[key][i])
    
    def check_groupfile(self):
        data = pd.read_csv(self._datadir+groupfile, header=0)
        for i in range(data.shape[0]):
            for key in groupfilekey:
                if data[key][i] not in self._listdata:
                    print("Nation:%s not found"%data[key][i])
                    self._listdata.append(data[key][i])
   
    def loaddata(self,build=False):
        if build:
            self.loadMatchFiles()
            self.check_groupfile()
           
            self._listdata.sort() 
            self._data = pd.DataFrame(self._listdata)
            self._data.columns=['Nation']
            self._data['ID'] = range(self._data.shape[0])
            #print(self._data)
            self._data.to_csv(self._datadir+csvfile)
        else:
            self._data = pd.read_csv(self._datadir+csvfile, header=0)
        return self._data

    def GetNationId(self,name):
        return self._data[self._data["Nation"] == name]["ID"].values[0]

    def GetNationName(self,id):
        return self._data[self._data["ID"] == id]["Nation"].values[0]

if __name__ == '__main__':
    #rdata = nations()
    #data = rdata.loaddata(build=True)

    rdata = nations()
    data = rdata.loaddata(build=False)

    print(data)

    print(rdata.GetNationId('Germany'))
    print(rdata.GetNationName(5))

