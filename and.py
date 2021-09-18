from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model

import pandas as pd
import numpy as np

def  main(data, eta, epoch, filename):

    df=pd.DataFrame(data)
    print(df)
    X,y= prepare_data(df)
    model=Perceptron(eta=eta, epoch=epoch)
    model.fit(X,y)
    _=model.total_loss() # dunmmy vaiable _

    save_model(model, filename=filename)

if __name__ == __name__:
    AND = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA= 0.3
    EPOCH=10

    main(data=AND, eta=ETA, epoch=EPOCH, filename="and.model")



