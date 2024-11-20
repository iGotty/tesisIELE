# environment/player.py

from environment.hand import Hand

class Player:
    def __init__(self):
        self.hands = []  # List of Hand instances

    def add_hand(self, hand):
        self.hands.append(hand)

    def reset_hands(self):
        self.hands = []

    def all_hands_finished(self):
        return all(hand.is_finished or hand.is_bust() for hand in self.hands)
