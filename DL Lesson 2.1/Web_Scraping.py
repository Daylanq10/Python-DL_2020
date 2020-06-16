import requests
from bs4 import BeautifulSoup

# PLACE CONTENTS OF HTML FILE IN BEAUTIFUL SOUP
url = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
soup = BeautifulSoup(url.content, "html.parser")

# OUTPUT <TITLE>
print("Title -> ", soup.title.string)

links = []

# FIND ALL HREF AND PLACE IN A LIST
for item in soup.find_all("a", href=True):
    print(item.get('href'))
    links.append(item.get('href'))

# OUTPUT LIST CONTENTS INTO A TEXT FILE
f_out = open("../../Desktop/Python-DL_2020/DL Lesson 2.1/output.txt", "a+")
for item in links:
    f_out.write(str(item))
    f_out.write('\n')


