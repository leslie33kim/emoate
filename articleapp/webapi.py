import requests

r = requests.get('http://158.247.204.237/projects/create/')

print(r.status_code)
print(r.text)