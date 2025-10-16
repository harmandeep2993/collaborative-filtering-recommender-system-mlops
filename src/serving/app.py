from fastapi import FastAPI, Query
from .predict import recommend_for_user, sample_user_id

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/sample_user")
def sample_user():
    return {"user_id": sample_user_id()}

@app.get("/recommend")
def recommend(user_id: int = Query(...), k: int = Query(10, ge=1, le=50)):
    return {"user_id": user_id, "k": k, "recommendations": recommend_for_user(user_id, k)}

@app.get("/")
def root():
    return {"status": "ok", "routes": ["/health", "/sample_user", "/recommend"]}
