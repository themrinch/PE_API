from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'model': 'facebook/bart-large-mnli'}


def test_predict_answer():
    response = client.post("/predict/",
                           json={"text": '''I applied for a visa invitation a few days/weeks ago and still haven't received it.
                           What should I do in this situation?'''})
    assert response.status_code == 200
    assert response.json() == '''I applied for a visa invitation a few days/weeks ago and still haven't received it. What should I do in this situation?
    There is nothing to worry about. The invitation is issued by the Ministry of Foreign Affairs of the Russian Federation and takes at least 30 days to be prepared. If you do not receive your invitation 40 days after submitting the application, please contact the Admissions Office at admission@urfu.ru.
    How do I extend my visa?
    Please visit the GUK-109 office at least six weeks prior to the expiration date of your current visa. You will be consulted on the visa extension procedures.'''
