import uvicorn

if __name__ == "__main__":
    uvicorn.run("api_model:app", host="127.0.0.1", port=5000, log_level="info")