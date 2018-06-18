import csv
import sys
import json

datafiles=['../data/wc2006.csv','../data/wc2010.csv','../data/wc2014.csv']
outdictfile='../data/nationdict.json'

def Usage(execname):
        print("Usage:")
        print(" %s [OutDictfile]"%(execname))
        print("")

def AddNation(nationlist,nation):
    if nation not in nationlist:
        nationlist.append(nation)

def GetDict(datafile,dic):
    with open(datafile) as f:
        reader = csv.reader(f,delimiter=';')
        rows=[row for row in reader]
        for row in rows:
        #for row in reader:
            #print(type(row))
            #print(len(row))
            AddNation(dic,row[2])
            AddNation(dic,row[3])

if __name__ == '__main__':
    if len(sys.argv) > 1:
        outdictfile = sys.argv[1]

    nationdict={}
    nations=[]
    for _,datafile in enumerate(datafiles):
        GetDict(datafile,nations)

    nations.sort()   
    for i,v in enumerate(nations):
        nationdict[v]=i
 
    with open(outdictfile,'w') as f:
        json.dump(nationdict,f,ensure_ascii=False,sort_keys=True,indent=4)  
        f.write('\n')
        f.close()
