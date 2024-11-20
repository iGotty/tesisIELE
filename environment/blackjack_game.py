# environment/blackjack_game.py

from environment.deck import Deck
from environment.dealer import Dealer
from environment.player import Player
from environment.hand import Hand
from environment.card import Card, Suit
from environment.rewards import calculate_reward
from environment.state_representation import get_state_representation

class BlackjackGameEnv:
    def __init__(self, deck_count=6, deck=None):
        self.deck = deck if deck is not None else Deck(deck_count)
        self.dealer = Dealer()
        self.player = Player()
        self.done = False
        self.true_count = 0
        self.max_splits = 3  # Límite de divisiones por mano

    def reset(self):
        # Evitar barajar si el mazo es predefinido
        if self.deck.get_penetration() >= 0.5 and not getattr(self.deck, 'predefined', False):
            self.deck.shuffle()
            self.true_count = 0
        else:
            self.true_count = self.calculate_true_count()

        self.dealer.reset_hand()
        self.player.reset_hands()
        self.done = False

        # Repartir cartas iniciales de manera normal
        initial_hand = Hand()
        initial_hand.add_card(self.deck.draw_card())
        initial_hand.add_card(self.deck.draw_card())
        self.player.add_hand(initial_hand)

        # Cartas para el dealer
        self.dealer.add_card(self.deck.draw_card())  # Dealer upcard
        self.dealer.add_card(self.deck.draw_card(), hidden=True)  # Dealer carta oculta

        self.update_true_count()
        return get_state_representation(self.player, self.dealer, self.true_count)


    def calculate_true_count(self):
        decks_remaining = self.deck.get_decks_remaining()
        running_count = self.deck.get_running_count()
        return running_count / decks_remaining if decks_remaining > 0 else 0

    def update_true_count(self):
        self.true_count = self.calculate_true_count()

    def is_action_valid(self, hand, action):
        if action == 'hit':
            return True
        elif action == 'stand':
            return True
        elif action == 'double':
            return len(hand.cards) == 2 and not hand.has_doubled
        elif action == 'split':
            return hand.can_split() and len(self.player.hands) < self.max_splits + 1
        else:
            return False




    def get_valid_actions(self, hand):
        valid_actions = ['hit', 'stand']  # Estas siempre son válidas
        if len(hand.cards) == 2 and not hand.has_doubled:
            valid_actions.append('double')
        if hand.can_split() and len(self.player.hands) < self.max_splits + 1:
            valid_actions.append('split')
        return valid_actions

    def step(self, action):
        # action es un diccionario {hand_index: action_name}
        # Si no se proporcionan acciones, asumimos "stand" para todas las manos activas
        if not action:
            action = {}
            for idx, hand in enumerate(self.player.hands):
                if not hand.is_finished and not hand.is_bust():
                    action[idx] = "stand"

        for idx, action_name in action.items():
            hand = self.player.hands[idx]

            if hand.is_finished:
                continue  # Omitir manos ya terminadas

            if hand.is_bust():
                hand.is_finished = True
                continue  # La mano ya está pasada, no hacemos nada más

            if not self.is_action_valid(hand, action_name):
                # Asignar una penalización y marcar la mano como terminada
                reward = -2  # Penalización por acción inválida
                hand.is_finished = True
                self.done = True
                return get_state_representation(self.player, self.dealer, self.true_count), reward, self.done

            # Procesar la acción
            if action_name == "hit":
                hand.add_card(self.deck.draw_card())
                self.update_true_count()
                if hand.is_bust():
                    hand.is_finished = True

            elif action_name == "stand":
                hand.is_finished = True

            elif action_name == "double":
                hand.double_bet()
                hand.add_card(self.deck.draw_card())
                self.update_true_count()
                hand.is_finished = True

            elif action_name == "split":
                new_hand = hand.split_hand()
                # Añadir una carta a cada mano después de dividir
                hand.add_card(self.deck.draw_card())
                new_hand.add_card(self.deck.draw_card())
                self.player.add_hand(new_hand)
                self.update_true_count()
                # No marcamos las manos como terminadas aquí

        # Después de procesar todas las acciones
        if self.player.all_hands_finished():
            self.dealer.play_hand(self.deck)
            self.done = True
            reward = calculate_reward(self.player, self.dealer, self.done)
        else:
            reward = 0  # Sin recompensa hasta que termine el juego

        return get_state_representation(self.player, self.dealer, self.true_count), reward, self.done