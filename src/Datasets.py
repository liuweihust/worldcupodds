import csv
import sys
import json


class dataset_worldcup():
    def __init__(self):
        self._nationsdict=None
        self._matches={}
        self._matchfiles={'2006':'../data/wc2006.csv','2010':'../data/wc2010.csv','2014':'../data/wc2014.csv'}
        self._nationdictfile='../data/nationdict.json'
        self._keys='date time host guest hwin deuce gwin hscore gscore comment'

    def GetNationName(self,nid):
        if self._nationsdict is None:
            return None
    
        for (key,value) in self._nationsdict.items():
            if value==nid:
                return key
        return None

    def GetNationCode(self,nation):
        if self._nationsdict is None:
            return None

        if self._nationsdict.has_key(nation):
            return self._nationsdict[nation]
        else:
            return None

    def loaddata(self):
        with open(self._nationdictfile,'r') as f:
            self._nationsdict=json.load(f)
            f.close()

        #print(self._matchfiles)
        for (key,dfile) in self._matchfiles.items():
            f = open(dfile,'r')
            reader = csv.reader(f,delimiter=';')
            rows=[row for row in reader]
            lists=[]
            for row in rows:
                match = {}
                match['date']=row[0]
                match['time']=row[1]
                match['host']=self.GetNationCode(row[2])
                match['guest']=self.GetNationCode(row[3])
                match['hwin']=float(row[4])
                match['deuce']=float(row[5])
                match['gwin']=float(row[6])
                match['hscore']=int(row[7])
                match['gscore']=int(row[8])
                match['comment']=row[9]
                lists.insert(0,match)
            self._matches[key]=lists
        #print(self._matches)

    def GetMatchDataByYear(self,year):
        return self._matches[year]

    def PrintNations(self):
        mystr=json.dumps(self._nationsdict,sort_keys=True,indent=4)
        print(mystr)

    def PrintMatches(self):
        for (year,yeardata) in self._matches.items():
            print("%s:"%(year))
            sys.stdout.write("%12s"%'date')
            sys.stdout.write("%6s"%'time')
            sys.stdout.write("%12s"%'host')
            sys.stdout.write("%12s"%'guest')
            sys.stdout.write("%5s"%'hwin')
            sys.stdout.write("%5s"%'draw')
            sys.stdout.write("%5s"%'gwin')
            sys.stdout.write("%8s"%'hscore')
            sys.stdout.write("%8s"%'gscore')
            sys.stdout.write("%5s"%'comt')
            sys.stdout.write("\n")

            for match in yeardata:
                sys.stdout.write("%12s"%match['date'])
                sys.stdout.write("%6s"%match['time'])
                sys.stdout.write("%12s"%self.GetNationName(match['host']))
                sys.stdout.write("%12s"%self.GetNationName(match['guest']))
                sys.stdout.write("%5.1f"%match['hwin'])
                sys.stdout.write("%5.1f"%match['deuce'])
                sys.stdout.write("%5.1f"%match['gwin'])
                sys.stdout.write("%8d"%match['hscore'])
                sys.stdout.write("%8d"%match['gscore'])
                sys.stdout.write("%5s"%match['comment'])
                sys.stdout.write("\n")

def Usage(execname):
        print("Usage:")
        print(" %s [OutDictfile]"%(execname))
        print("")

if __name__ == '__main__':
    data = dataset_worldcup()
    data.loaddata()
    #data.PrintNations()
    data.PrintMatches()
