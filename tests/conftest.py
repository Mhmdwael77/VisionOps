from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base, get_db
from backend.app.main import app


@pytest.fixture
def database(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    yield factory
    engine.dispose()


@pytest.fixture
def api_client(database, tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.main.EXPERIMENT_LOCK", tmp_path/"experiment.lock")
    isolated = FastAPI()
    isolated.include_router(app.router)
    def session():
        with database() as db:
            yield db
    isolated.dependency_overrides[get_db] = session
    with TestClient(isolated) as client:
        yield client
