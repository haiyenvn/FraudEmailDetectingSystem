from bs4 import BeautifulSoup
from urllib import request

url = 'https://monkey.org/~jose/phishing/phishing-2020'

soup = BeautifulSoup(request.urlopen(url).read(), "lxml")

with open("phishing-2020_2.mbox", "a") as f:
    print(soup.get_text(), file=f)