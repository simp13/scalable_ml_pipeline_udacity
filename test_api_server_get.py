def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World!"}
