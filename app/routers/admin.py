from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings
from app.jobs.retrain_model import retrain_model_job

router = APIRouter()

# Define API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key")


def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == settings.ADMIN_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )


@router.post("/retrain", status_code=status.HTTP_200_OK)
async def retrain_model(api_key: str = Depends(get_api_key)):
    try:
        retrain_model_job()
        return {"message": "Model retraining initiated successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retraining failed: {e}",
        )
