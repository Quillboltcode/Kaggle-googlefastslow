import sqlite3

class News:
    def __init__(self) -> None:# contruct php
        self.name = "News"
        self.image = "news.png"



array_of_news = [News(),News(),News()]

def get_news(id):
    # return $array_of_news[id]
    return array_of_news[id]

def get_news_from_db():
    db = sqlite3.connect('news.db')
    # get data from database 
    cursor = db.cursor()
    cursor.execute("SELECT * FROM news limit 3")
    data = cursor.fetchall()

    for i in range(len(data)):
        array_of_news[i].name = data[i][1]
        array_of_news[i].image = data[i][2]

def get_next_news(id):
    return array_of_news[(id+1)
                         ]