import uvicorn
import pickle
from fastapi import FastAPI
import sklearn

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
    print(model.predict([[0, 1, 2]]))
    
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hey look its Working'}
            
@app.get('/predict')
def predict(feature_1: int = 0, feature_2: int = 0, feature_3: int = 0):
    return {'output': model.predict([[feature_1, feature_2, feature_3]]).tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
