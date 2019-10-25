import requests
import numpy as np

class Card:
    def __init__(self, name, group, label):
        self.name = name
        self.group = group
        self.board = self.group.board
        self.id = None
        self.label = label

    def make(self):
        url = "https://api.trello.com/1/cards"
        query = {"idList": self.group.id, "idLabels": self.board.labels[self.label], "name": self.name, "key": self.board.key, "token": self.board.token}
        response = requests.request("POST", url, params=query)
        self.id = response.json()['id']

class Group:
    def __init__(self, name, board, cards):
        self.name = name
        self.board = board
        self.cards = self.build_cards(cards)
        self.id = None

    def build_cards(self, cards):
        card_list = []
        for card in cards:
            card_list.append(Card(card['name'], self, card['label']))
        return card_list

    def make(self):
        url = "https://api.trello.com/1/lists"
        query = {"name": self.name, "idBoard": self.board.id, "key": self.board.key, "token": self.board.token}
        response = requests.request("POST", url, params=query).json()
        self.id = response['id']
        for card in self.cards:
            card.make()

class Board:
    def __init__(self, key, token, name, groups):
        self.name = name
        self.labels = self.parse_labels(groups)
        self.groups = self.build_groups(groups)
        self.key = key
        self.token = token
        self.id = None
        self.url = None

    def build_groups(self, groups):
        group_list = []
        for group in groups:
            group_list.append(Group(group['name'], self, group['cards']))
        return group_list

    def make(self):
        url = 'https://api.trello.com/1/boards'
        query = {'name': self.name, 'defaultLabels': 'false', 'defaultLists': 'false', 'key': self.key, 'token': self.token}
        response = requests.request("POST", url, params = query).json()
        self.id = response['id']
        self.make_labels()
        for group in reversed(self.groups):
            group.make()
        self.url = self.board_url()

    def parse_labels(self, groups):
        labels = set()
        for group in groups:
            for card in group['cards']:
                labels.add(card['label'])
        return labels

    def make_labels(self):
        labels = {}
        for label in self.labels:
            url = "https://api.trello.com/1/labels"
            query = {"name": "", "color": label, "idBoard": self.id, "key": self.key, "token": self.token}
            response = requests.request("POST", url, params=query)
            labels[label] = response.json()['id']
        self.labels = labels

    def invite(self, name, email):
        url = "https://api.trello.com/1/boards/{}/members".format(self.id)
        query = {"email": email, "key": self.key, "token": self.token}
        payload = "{\"fullName\":\"" + name + "\"}"
        headers = {
            'type': "admin",
            'content-type': "application/json"
            }
        response = requests.request("PUT", url, data=payload, headers=headers, params=query)

    def board_url(self, short=True):
        url = "https://api.trello.com/1/boards/{}".format(self.id)
        query = {"key": self.key, "token": self.token}
        query["fields"] = "shortUrl" if short else "url"
        response = requests.request("GET", url, params=query).json()
        return response["shortUrl"] if short else response["url"]

def authorise(self, key):
        url = "https://api.trello.com/1/authorize"
        query = {"callback_method": "", "return_url": myurl, "scope": "read,write", "expiration": "1hour", "name": "sticky2trello", "key": key, "response_type": "token"}
        response = requests.request("POST", url, params=query)

if __name__ == "__main__":
    groups = [{'name': 'group 1', 'cards': [{'name':'a card', 'label': 'pink'}, {'name': 'another card', 'label': 'green'}]},
        {'name': 'group 2', 'cards': [{'name': 'a 2nd list card', 'label': 'pink'}, {'name': 'a 2nd 2nd list card', 'label': 'yellow'}]}]
    
    board = Board('testBoard', groups)
    print('board built')
    board.make()
