# environment/hand.py

from environment.card import Card

class Hand:
    def __init__(self, bet_size=1):
        self.cards = []
        self.bet_size = bet_size
        self.is_insured = False
        self.is_finished = False  # Indicates if the hand has been played
        self.has_doubled = False

    def add_card(self, card):
        self.cards.append(card)

    def get_value(self):
        total = sum(card.get_value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 1)
        while total > 21 and aces:
            total -= 10  # Counting an Ace as 1 instead of 11
            aces -= 1
        return total

    def is_blackjack(self):
        return len(self.cards) == 2 and self.get_value() == 21

    def is_bust(self):
        return self.get_value() > 21

    def can_split(self):
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    def split_hand(self):
        split_card = self.cards.pop()
        new_hand = Hand(bet_size=self.bet_size)
        new_hand.add_card(split_card)
        return new_hand

    def double_bet(self):
        if not self.has_doubled and len(self.cards) == 2:
            self.bet_size *= 2
            self.has_doubled = True

    def reset(self):
        self.cards = []
        self.bet_size = 1
        self.is_insured = False
        self.is_finished = False
        self.has_doubled = False

    def is_soft(self):
        aces = sum(1 for card in self.cards if card.rank == 1)
        total_without_aces = sum(card.get_value() for card in self.cards if card.rank != 1)
        if aces > 0 and total_without_aces + 11 + (aces - 1) <= 21:
            return True
        else:
            return False

    def is_pair(self):
        # Returns True if the hand can be split (both cards have the same rank)
        return self.can_split()
