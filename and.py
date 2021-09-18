from utils.model import Perceptron



AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y": [0,0,0,1],
}

df=pd.DataFrame(AND)

df

X,y= prepare_data(df)
ETA= 0.3
EPOCH=10
model=Perceptron(eta=ETA, epoch=EPOCH)
model.fit(X,y)
_=model.total_loss()
