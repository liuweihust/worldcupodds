import csv
import sys
import json
import pandas as pd
import numpy as np
import NationsDict

groupfile='group.csv'
groupfilekey=['M1','M2','M3','M4']
yearkey='Year'
groupkey='Group'
groupfile='group.csv'
nationfile='nations.csv'

class Groups():
    def __init__(self,data_dir='../data/'):
        self._data={}
        self._datadir=data_dir
       
    def LoadFile(self):
        data = pd.read_csv(self._datadir+groupfile, header=0)
        for i in range(data.shape[0]):
            year = data[yearkey][i]
            group = data[groupkey][i]
            members=[]
            for key in groupfilekey:
                members.append(data[key][i])

            #groupdata={}
            #groupdata[group]=members
            if not self._data.has_key(year):
                self._data[year]={}

            self._data[year][group]=members

    def GetGroupNames(self,year=2018):
        return self._data[year].keys()
    
    def GetGroupMem(self,year,group):
        return self._data[year][group]
 
    def GetData(self):
        return self._data
    
    def GetGroup(self,name,year=2018):
        if not self._data.has_key(year):
            return None

        for key in self._data[year]:
            if name in self._data[year][key]:
                return key

        return None 
 
    def InSameGroup(self,m1,m2,year=2018):
        g1 = self.GetGroup(m1,year)
        g2 = self.GetGroup(m2,year)
        if g1 is None or g2 is None:
            return None
        else:
            return g1==g2

if __name__ == '__main__':
    rdata = Groups()
    rdata.LoadFile()
    data = rdata.GetData()
    print(data)

    print(rdata.GetGroupNames(2014))
    print(rdata.GetGroup('Germany',2014))
    print(rdata.GetGroupMem(2014,'G'))
    print(rdata.InSameGroup('Italy','Brazil',2014))
    print(rdata.InSameGroup('Italy','England',2014))

