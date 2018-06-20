import csv
import sys
import json
import numpy as np

def VecScore2Res90(rec):
    if rec['comment'] != '':
        return [0.,1.,0.]
    elif rec['hscore'] > rec['gscore']:
        return [1.,0.,0.]
    elif rec['hscore'] < rec['gscore']:
        return [0.,0.,1.]
    else:
        return [0.,1.,0.]

def Score2Res90(rec):
    #[HostWin,Deuce,GuestWin]=[2,1,0]
    if rec['comment'] != '':
        return 1
    elif rec['hscore'] > rec['gscore']:
        return 2
    elif rec['hscore'] < rec['gscore']:
        return 0
    else:
        return 1

def offer2Res90(rec):
    rsum=0.
    rsum=rec['hwin']+rec['deuce']+rec['gwin']
    return [rec['hwin']/rsum,rec['deuce']/rsum,rec['gwin']/rsum]

class dataset_worldcup():
    def __init__(self,mode='WDL',batch_size=20):
        self._nationsdict=None
        self._datadir='../data/'
        self._trainfiles={'2006':'wc2006.csv','2010':'wc2010.csv','2014':'wc2014.csv'}
        self._testfiles={'2018':'wc2018.csv'}
        self._nationdictfile='nationdict.json'
        self._keys='date time host guest hwin deuce gwin hscore gscore comment'
        self._mode=mode

        self._batch_size=batch_size
        self._batchindex=0

        self._trainmatches={}
        self._traindata=[]
        self._matchnum=0

        self._testmatches={}
        self._testdata=[]

    def GetInputShape(self):
        return 3

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

    def loaddata(self,datadir='../data/'):
        with open(self._datadir+self._nationdictfile,'r') as f:
            self._nationsdict=json.load(f)
            f.close()

        lista=[]
        #print(self._trainfiles)
        for (key,dfile) in self._trainfiles.items():
            f = open(self._datadir+dfile,'r')
            reader = csv.reader(f)
            rows=[row for row in reader]
            lists=[]
            for row in rows:
                match = {}
                res=[]
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
    
                if self._mode == 'WDL':
                    #res.extend(offer2Res90(match))
                    #res.extend(Score2Res90(match))

                    res.extend([match['hwin'],match['deuce'],match['gwin']])
                    res.append( float(Score2Res90(match)) )
                    lista.append(res)

            self._trainmatches[key]=lists
        self._traindata=np.array(lista)
        self._matchnum=self._traindata.shape[0]

        for (key,dfile) in self._testfiles.items():
            f = open(self._datadir+dfile,'r')
            reader = csv.reader(f)
            rows=[row for row in reader]
            lists=[]
            for row in rows:
                match = {}
                res=[]
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
    
                if self._mode == 'WDL':
                    #res.extend(offer2Res90(match))
                    #res.extend(Score2Res90(match))

                    res.extend([match['hwin'],match['deuce'],match['gwin']])
                    res.append( float(Score2Res90(match)) )
                    lista.append(res)

            self._testmatches[key]=lists
            self._testdata=np.array(lista)

    def GetMatchDataByYear(self,year):
        return self._trainmatches[year]

    def PrintNations(self):
        mystr=json.dumps(self._nationsdict,sort_keys=True,indent=4)
        print(mystr)

    def PrintMatches(self):
        for (year,yeardata) in self._trainmatches.items():
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

        for (year,yeardata) in self._testmatches.items():
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

    def GetNextbatch(self):
        if self._batch_size+self._batchindex < self._matchnum:
            a = self._traindata[self._batchindex:self._batch_size+self._batchindex,0:3]
            b = self._traindata[self._batchindex:self._batch_size+self._batchindex,3]
            self._batchindex += self._batch_size
            if self._batchindex >= self._matchnum:
                self._batchindex=0
            return a,b
        else:
            a=self._traindata[self._batchindex:,0:3]
            b=self._traindata[self._batchindex:,3]
            if self._batch_size + self._batchindex > self._matchnum:
                aa = self._traindata[0:self._batch_size + self._batchindex - self._matchnum,0:3]
                bb = self._traindata[0:self._batch_size + self._batchindex - self._matchnum,3]
                a = np.concatenate(a,aa)
                b = np.concatenate(b,bb)
            self._batchindex = 0
            return a,b

    def GetTestData(self):
        return self._testdata[:,0:3],self._testdata[:,3]

def Usage(execname):
        print("Usage:")
        print(" %s [OutDictfile]"%(execname))
        print("")

if __name__ == '__main__':
    data = dataset_worldcup()
    data.loaddata()
    data.PrintNations()
    data.PrintMatches()

    """
    for _ in range(20):
        a,b = data.GetNextbatch()
        print(a,b)
    """
