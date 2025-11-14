import requests

url = "https://api.esa.io/v1/teams/cs18a/posts"
headers = {"Authorization": "Bearer RmifgK844jPSHmqpe8Y2u_clXGJeiGIg31ZIt-ZIzpg"}

r = requests.get(url, headers=headers)
print(r.status_code)
print(r.json())

url = "https://api.esa.io/v1/teams"
headers = {"Authorization": "Bearer RmifgK844jPSHmqpe8Y2u_clXGJeiGIg31ZIt-ZIzpg"}
r = requests.get(url, headers=headers)
print(r.status_code)
print(r.text)
