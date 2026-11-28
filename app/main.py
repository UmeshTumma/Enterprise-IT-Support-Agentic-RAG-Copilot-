
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.routes import router
from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.services.audit import init_db

logger = logging.getLogger(__name__)

configure_logging()
settings = get_settings()
init_db()


app = FastAPI(title=settings.app_name, version="1.0.0")

app.include_router(router)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        template = templates.get_template("index.html")

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "app_name": settings.app_name
            }
        )

    except Exception as e:
        logger.exception("Homepage template error")
        return HTMLResponse(
            content=f"Homepage error: {str(e)}",
            status_code=500
        )