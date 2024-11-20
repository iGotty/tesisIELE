# environment/dealer.py

from environment.hand import Hand
from environment.card import Card

class Dealer:
    def __init__(self):
        self.hand = Hand()
        self.upcard = None  # The dealer's visible card

    def reset_hand(self):
        self.hand = Hand()
        self.upcard = None

    def add_card(self, card, hidden=False):
        self.hand.add_card(card)
        if not hidden:
            self.upcard = card

    def play_hand(self, deck):
        while self.hand.get_value() < 17:
            self.hand.add_card(deck.draw_card())

    def get_upcard_value(self):
        return self.upcard.get_value() if self.upcard else 0

    def is_bust(self):
        return self.hand.is_bust()

    def get_value(self):
        return self.hand.get_value()
