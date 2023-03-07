from fastapi import FastAPI, File, UploadFile
import uvicorn
from img_pred import Imagepredict

app = FastAPI()

@app.get('/index')
def index():
    return 'Hello'

@app.get('/index/{name}')
def name(name:str):
    return f'Hello {name}!'

@app.post('/predict')
async def img_predict(file: UploadFile = File(...)):
    image = Imagepredict.read_image(await file.read())
    image = Imagepredict.preprocess(image)
    predictions = Imagepredict.predict(image)
    return predictions


if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8020)