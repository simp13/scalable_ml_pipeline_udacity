import requests


data = {
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
    }
r = requests.post('https://udacity-scalable-ml-pipeline.herokuapp.com/', json=data)

assert r.status_code == 200

print("Response code: {}".format(r.status_code))
print("Response body: {}".format(r.json()))