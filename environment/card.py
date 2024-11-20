# environment/card.py

from enum import Enum

class Suit(Enum):
    CLUBS = "Clubs"
    DIAMONDS = "Diamonds"
    HEARTS = "Hearts"
    SPADES = "Spades"

class Card:
    def __init__(self, rank, suit):
        self.rank = rank  # 1 to 13
        self.suit = suit  # Instance of Suit Enum

    def get_value(self):
        if self.rank > 10:
            return 10
        elif self.rank == 1:
            return 11  # Ace initially counts as 11
        else:
            return self.rank

    def __str__(self):
        rank_name = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(self.rank, str(self.rank))
        return f"{rank_name} of {self.suit.value}"

    def __repr__(self):
        return self.__str__()
