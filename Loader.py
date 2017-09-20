import pandas as pd

class Loader(object):
    def __init__(self,trainfile,propertyfile):
        
        self.train=pd.read_csv(trainfile,parse_date=['transactiondate'])
        self.prop=pd.read_csv(propertyfile,chunksize=1000)
        self.prop_chunks=[]
        for chunk in self.prop:
            self.prop_chunks.append(chunk)
        self.prop=pd.concat(self.prop_chunks,ignore_index=True)
