from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}
@app.post("/predict_spending/")
async def predict_spending(data: dict):
    return {"message": "Spending prediction received", "data": data}