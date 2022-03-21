import pytest
from fastapi.testclient import TestClient
from api_server import app


@pytest.fixture
def client(scope='session'):
    """
    Get test client
    """
    api_client = TestClient(app)
    return api_client
