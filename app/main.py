###########################################################
#      __                   _                        _    #
#     / /____  ____  ____  (_)____      ____ _____  (_)   #
#    / __/ _ \/ __ \/ __ \/ / ___/_____/ __ `/ __ \/ /    #
#   / /_/  __/ / / / / / / (__  )_____/ /_/ / /_/ / /     #
#   \__/\___/_/ /_/_/ /_/_/____/      \__,_/ .___/_/      #
#                                         /_/             #
###########################################################


from typing import Dict
from fastapi import FastAPI
from app.core.config import get_api_settings
from app.routes.predict import PredictRouter
from app.routes.model import ModelRouter


settings = get_api_settings()
TITLE = settings.title
CONTACTS = settings.contacts
URL_DOC = settings.redoc_url
URL_SWAGGER = settings.docs_url


app = FastAPI(
    title = TITLE,
    contacts = CONTACTS,
    redoc_url = URL_DOC,
    docs_url = URL_SWAGGER
)

app.include_router(PredictRouter)
app.include_router(ModelRouter)


@app.get('/')
async def about() -> Dict[str, str]:
    """Give information about the API.

    Returns:
        Dict[str, str]: With shape :
    `
    {"app_title": <TITLE>, "app_contacts": <CONTACTS>, "api_url_doc": <URL_DOC>, "api_url_swagger": <URL_SWAGGER>}
    `
    """
    return {
        "app_title": TITLE,
        "app_contacts": CONTACTS,
        "api_url_doc": URL_DOC,
        "api_url_swagger": URL_SWAGGER
    }
