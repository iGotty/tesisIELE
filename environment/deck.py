# environment/deck.py

import random
from environment.card import Card, Suit

class Deck:
    def __init__(self, num_decks=6, cards=None):
        self.num_decks = num_decks
        self.running_count = 0
        if cards is not None:
            self.cards = cards  # No invertimos el orden
            self.predefined = True
        else:
            self.build_deck()

    def draw_card(self):
        if not self.cards:
            self.build_deck()
        if getattr(self, 'predefined', False):
            card = self.cards.pop(0)  # Extraemos del frente
        else:
            card = self.cards.pop()   # Extraemos del final
        self.update_running_count(card)
        return card
    
    def build_deck(self):
        self.cards = [Card(rank, suit) for _ in range(self.num_decks)
                      for suit in Suit for rank in range(1, 14)]
        random.shuffle(self.cards)
        self.running_count = 0
        
    def update_running_count(self, card):
        if 2 <= card.rank <= 6:
            self.running_count += 1
        elif card.rank in {1, 10, 11, 12, 13}:
            self.running_count -= 1

    def get_running_count(self):
        return self.running_count

    def get_penetration(self):
        return 1 - (len(self.cards) / (self.num_decks * 52))

    def get_decks_remaining(self):
        return max(1, len(self.cards) / 52)

    def shuffle(self):
        self.build_deck()
