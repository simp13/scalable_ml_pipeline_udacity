def test_post_greater_than_50K(client):
    r = client.post("/", json={
        "age": 42,
        "workclass": "Private",
        "fnlgt": 159449,
        "education": "Bachelors",
        "educationNum": 13,
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capitalGain": 5178,
        "capitalLoss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"result": ">50K"}


def test_post_lower_than_50K(client):
    r = client.post("/", json={
        "age": 17,
        "workclass": "Private",
        "fnlgt": 102726,
        "education": "12th",
        "educationNum": 8,
        "maritalStatus": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capitalGain": 0,
        "capitalLoss": 0,
        "hoursPerWeek": 16,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"result": "<=50K"}


def test_post_wrong(client):
    r = client.post("/", json={
        "age": 37,
        "workclass": "Private",
        "fnlgt": 284582,
        "education": "Masters",
        "educationNum": 14,
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capitalGain": 0,
        "capitalLoss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "Test Country"
    })
    assert r.status_code == 422
