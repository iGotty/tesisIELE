# tests/test_blackjack_game.py

import unittest
from environment.blackjack_game import BlackjackGameEnv
from environment.card import Card, Suit
from environment.hand import Hand
from environment.deck import Deck


class TestBlackjackGameEnv(unittest.TestCase):
    def setUp(self):
        self.env = BlackjackGameEnv()

    def test_initial_state(self):
        state = self.env.reset()
        self.assertIsInstance(state, dict)
        self.assertIn('player_value', state)
        self.assertIn('dealer_upcard', state)
        self.assertIn('true_count', state)
        self.assertIn('hand_type', state)
        self.assertGreaterEqual(state['player_value'], 2)
        self.assertLessEqual(state['player_value'], 21)
        self.assertGreaterEqual(state['dealer_upcard'], 1)
        self.assertLessEqual(state['dealer_upcard'], 11)
        self.assertIn(state['hand_type'], [0, 1, 2])  # Verificamos que hand_type sea válido


    def test_hit_action(self):
        # Predefinimos las cartas para asegurar que el jugador no se pase
        # Orden de reparto: Jugador, Dealer, Jugador, Dealer
        cards = [
            Card(2, Suit.HEARTS),    # Jugador carta 1
            Card(5, Suit.CLUBS),     # Dealer carta 1 (upcard)
            Card(3, Suit.DIAMONDS),  # Jugador carta 2
            Card(10, Suit.SPADES),   # Dealer carta 2 (oculta)
            Card(4, Suit.CLUBS),     # Carta para el "hit" del jugador
            # Cartas adicionales si son necesarias
        ]
        self.env = BlackjackGameEnv(deck=Deck(cards=cards))
        self.env.reset()
        initial_hand_value = self.env.player.hands[0].get_value()
        actions = {0: 'hit'}
        state, reward, done = self.env.step(actions)
        new_hand_value = self.env.player.hands[0].get_value()
        self.assertFalse(done)
        self.assertGreaterEqual(new_hand_value, initial_hand_value)
        self.assertEqual(new_hand_value, state['player_value'])
        # Verificamos que el jugador no se ha pasado
        self.assertFalse(self.env.player.hands[0].is_bust())




    def test_stand_action(self):
        self.env.reset()
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        # Dado que todas las manos se han plantado, el juego debería haber terminado
        self.assertTrue(done)
        self.assertIsInstance(reward, int)

    def test_double_action(self):
        self.env.reset()
        hand = self.env.player.hands[0]
        # Forzamos una mano adecuada para doblar
        hand.cards = [Card(5, Suit.HEARTS), Card(6, Suit.SPADES)]  # Total 11
        actions = {0: 'double'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(hand.has_doubled)
        self.assertTrue(hand.is_finished)
        self.assertTrue(done)  # Ahora debería ser True



    def test_split_action(self):
        self.env.reset()
        hand = self.env.player.hands[0]
        # Forzamos que la mano sea divisible
        hand.cards = [Card(8, Suit.HEARTS), Card(8, Suit.SPADES)]
        actions = {0: 'split'}
        state, reward, done = self.env.step(actions)
        self.assertEqual(len(self.env.player.hands), 2)
        self.assertFalse(done)
        self.assertEqual(len(self.env.player.hands[0].cards), 2)
        self.assertEqual(len(self.env.player.hands[1].cards), 2)

    def test_split_then_hit(self):
        # Cartas repartidas en este orden: Jugador, Jugador, Dealer, Dealer
        cards = [
            Card(8, Suit.CLUBS),     # Jugador carta 1
            Card(8, Suit.DIAMONDS),  # Jugador carta 2
            Card(5, Suit.HEARTS),    # Dealer carta 1 (upcard)
            Card(10, Suit.SPADES),   # Dealer carta 2 (oculta)
            Card(2, Suit.CLUBS),     # Carta para hit en mano 0
            # Cartas adicionales si es necesario
        ]
        self.env = BlackjackGameEnv(deck=Deck(cards=cards))
        self.env.reset()
        actions = {0: 'split'}
        state, reward, done = self.env.step(actions)
        # Verificamos los índices de las manos después del split
        self.assertEqual(len(self.env.player.hands), 2)
        actions = {0: 'hit', 1: 'stand'}
        state, reward, done = self.env.step(actions)
        # Verificamos el estado de cada mano
        hand0 = self.env.player.hands[0]
        hand1 = self.env.player.hands[1]
        self.assertFalse(hand0.is_finished)
        self.assertTrue(hand1.is_finished)
        # El juego no debería haber terminado si hay manos activas
        self.assertFalse(done)





    def test_multiple_hits(self):
        # Simula un escenario donde el jugador hace varios "hits" sin pasarse
        cards = [
            Card(2, Suit.HEARTS),    # Jugador carta 1
            Card(3, Suit.SPADES),    # Jugador carta 2
            Card(5, Suit.CLUBS),     # Dealer carta 1 (upcard)
            Card(10, Suit.DIAMONDS), # Dealer carta 2 (oculta)
            Card(4, Suit.CLUBS),     # Carta para hit 1
            Card(5, Suit.HEARTS),    # Carta para hit 2
        ]
        self.env = BlackjackGameEnv(deck=Deck(cards=cards))
        self.env.reset()
        hand = self.env.player.hands[0]
        actions = {0: 'hit'}
        state, reward, done = self.env.step(actions)
        actions = {0: 'hit'}
        state, reward, done = self.env.step(actions)
        # Verificamos que la mano no está "bust"
        self.assertFalse(hand.is_bust())
        # Verificamos que el valor de la mano es 14 (2 + 3 + 4 + 5)
        self.assertEqual(hand.get_value(), 14)





    def test_bust_hand(self):
        # Orden de reparto: Jugador, Dealer, Jugador, Dealer
        cards = [
            Card(10, Suit.HEARTS),     # Jugador carta 1
            Card(9, Suit.DIAMONDS),    # Dealer carta 1 (upcard)
            Card(10, Suit.SPADES),     # Jugador carta 2
            Card(7, Suit.CLUBS),       # Dealer carta 2 (oculta)
            Card(5, Suit.DIAMONDS),    # Jugador hit (para bust)
        ]
        self.env = BlackjackGameEnv(deck=Deck(cards=cards))
        self.env.reset()
        hand = self.env.player.hands[0]
        actions = {0: 'hit'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(hand.is_bust())
        self.assertTrue(hand.is_finished)
        self.assertTrue(done)
        self.assertLess(reward, 0)



    def test_dealer_play(self):
        self.env.reset()
        # El jugador se planta inmediatamente
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        # El dealer debería haber jugado su mano
        dealer_value = self.env.dealer.get_value()
        self.assertGreaterEqual(dealer_value, 17)

    def test_reward_win(self):
        self.env.reset()
        # Forzamos que la mano del jugador sea mayor que la del dealer
        player_hand = self.env.player.hands[0]
        dealer_hand = self.env.dealer.hand
        player_hand.cards = [Card(10, Suit.HEARTS), Card(9, Suit.SPADES)]  # 19
        dealer_hand.cards = [Card(10, Suit.CLUBS), Card(6, Suit.DIAMONDS)]  # 16

        # Mockeamos el método play_hand del dealer
        def mock_play_hand(deck):
            pass  # El dealer no pide más cartas
        self.env.dealer.play_hand = mock_play_hand

        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        self.assertGreater(reward, 0)


    def test_reward_loss(self):
        self.env.reset()
        # Forzamos que la mano del jugador sea menor que la del dealer
        player_hand = self.env.player.hands[0]
        dealer_hand = self.env.dealer.hand
        player_hand.cards = [Card(10, Suit.HEARTS), Card(6, Suit.SPADES)]  # 16
        dealer_hand.cards = [Card(10, Suit.CLUBS), Card(9, Suit.DIAMONDS)]  # 19
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        self.assertLess(reward, 0)

    def test_reward_push(self):
        self.env.reset()
        # Forzamos que la mano del jugador sea igual a la del dealer
        player_hand = self.env.player.hands[0]
        dealer_hand = self.env.dealer.hand
        player_hand.cards = [Card(10, Suit.HEARTS), Card(8, Suit.SPADES)]  # 18
        dealer_hand.cards = [Card(10, Suit.CLUBS), Card(8, Suit.DIAMONDS)]  # 18
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        self.assertEqual(reward, 0)

    def test_true_count_update(self):
        self.env.reset()
        initial_true_count = self.env.true_count
        # Simulamos extracción de cartas para cambiar el conteo
        self.env.deck.running_count = 10
        self.env.update_true_count()
        self.assertNotEqual(self.env.true_count, initial_true_count)
        expected_true_count = self.env.deck.running_count / self.env.deck.get_decks_remaining()
        self.assertEqual(self.env.true_count, expected_true_count)

    def test_deck_shuffle_on_penetration(self):
        self.env = BlackjackGameEnv(deck_count=6)
        self.env.deck.cards = self.env.deck.cards[:100]  # Simular penetración alta
        self.env.reset()
        # Después de reset, el mazo debe haberse barajado si la penetración es alta
        expected_cards_remaining = 312 - 4  # 6 mazos * 52 cartas - 4 cartas repartidas
        self.assertEqual(len(self.env.deck.cards), expected_cards_remaining)


    def test_max_splits(self):
        # Orden de reparto: Jugador, Jugador, Dealer, Dealer
        cards = [
            Card(8, Suit.HEARTS),    # Jugador carta 1
            Card(8, Suit.SPADES),    # Jugador carta 2
            Card(5, Suit.CLUBS),     # Dealer carta 1 (upcard)
            Card(10, Suit.DIAMONDS), # Dealer carta 2 (oculta)
            # Cartas para splits
            Card(8, Suit.DIAMONDS),  # Para mano 1
            Card(8, Suit.CLUBS),     # Para mano 2
            Card(8, Suit.HEARTS),    # Para mano 3
            Card(8, Suit.SPADES),    # Para mano 4
            # Cartas adicionales si es necesario
        ]
        self.env = BlackjackGameEnv(deck=Deck(cards=cards))
        self.env.reset()
        actions = {}
        while True:
            actions = {}
            for idx, hand in enumerate(self.env.player.hands):
                if hand.can_split() and len(self.env.player.hands) < self.env.max_splits + 1:
                    actions[idx] = 'split'
            if not actions:
                break
            state, reward, done = self.env.step(actions)
        # Después del número máximo de splits, no debería ser posible dividir más
        expected_hands = self.env.max_splits + 1  # Mano original + max_splits
        self.assertEqual(len(self.env.player.hands), expected_hands)
        self.assertFalse(done)
        # Terminamos el juego
        while not done:
            actions = {idx: 'stand' for idx, hand in enumerate(self.env.player.hands) if not hand.is_finished}
            state, reward, done = self.env.step(actions)
        self.assertTrue(done)


    def test_multiple_hands_play(self):
        # Definimos las cartas para el mazo en el orden que se repartirán
        cards = [
            Card(8, Suit.HEARTS),    # Jugador carta 1
            Card(8, Suit.SPADES),    # Jugador carta 2
            Card(5, Suit.CLUBS),     # Dealer carta 1 (upcard)
            Card(10, Suit.DIAMONDS), # Dealer carta 2 (oculta)
            Card(2, Suit.CLUBS),     # Carta añadida a mano 0 después del split
            Card(3, Suit.HEARTS),    # Carta añadida a mano 1 después del split
            Card(4, Suit.SPADES),    # Carta para 'hit' en mano 0
            # Cartas adicionales si es necesario
        ]
        self.env = BlackjackGameEnv(deck=Deck(cards=cards))
        self.env.reset()
        actions = {0: 'split'}
        state, reward, done = self.env.step(actions)
        # Después del split y de añadir cartas, las manos deben tener dos cartas cada una
        self.assertEqual(len(self.env.player.hands), 2)
        hand0 = self.env.player.hands[0]
        hand1 = self.env.player.hands[1]
        self.assertEqual(len(hand0.cards), 2)
        self.assertEqual(len(hand1.cards), 2)
        # Verificamos que ninguna mano está terminada aún
        self.assertFalse(hand0.is_finished)
        self.assertFalse(hand1.is_finished)
        # Jugamos cada mano de manera diferente
        actions = {0: 'hit', 1: 'stand'}
        state, reward, done = self.env.step(actions)
        hand0 = self.env.player.hands[0]
        hand1 = self.env.player.hands[1]
        # Verificamos que la mano 1 ha terminado (se plantó)
        self.assertTrue(hand1.is_finished)
        # Verificamos que la mano 0 no ha terminado
        self.assertFalse(hand0.is_finished)
        # Verificamos que la mano 0 no se ha pasado
        self.assertFalse(hand0.is_bust())
        # Continuamos jugando la mano 0
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        hand0 = self.env.player.hands[0]
        # Verificamos que la mano 0 ahora ha terminado
        self.assertTrue(hand0.is_finished)
        # El juego debe haber terminado ya que todas las manos han finalizado
        self.assertTrue(done)





    def test_hand_finished_flag(self):
        self.env.reset()
        hand = self.env.player.hands[0]
        self.assertFalse(hand.is_finished)
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(hand.is_finished)

    def test_action_default_to_stand(self):
        self.env.reset()
        # No se proporciona acción, debería por defecto plantarse
        actions = {}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)

    def test_invalid_action(self):
        self.env.reset()
        actions = {0: 'invalid_action'}
        state, reward, done = self.env.step(actions)
        # Verificamos que el juego ha terminado
        self.assertTrue(done)
        # Verificamos que se ha asignado una penalización
        self.assertEqual(reward, -2)  # O el valor que hayas asignado como penalización en el entorno


    def test_split_aces(self):
        self.env.reset()
        hand = self.env.player.hands[0]
        # Forzamos un par de ases
        hand.cards = [Card(1, Suit.HEARTS), Card(1, Suit.SPADES)]
        actions = {0: 'split'}
        state, reward, done = self.env.step(actions)
        # Asumimos que permitimos pedir carta después de dividir ases
        actions = {0: 'hit', 1: 'hit'}
        state, reward, done = self.env.step(actions)
        self.assertFalse(done)
        # Nos plantamos en ambas manos
        actions = {0: 'stand', 1: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)

    def test_double_after_split(self):
        self.env.reset()
        hand = self.env.player.hands[0]
        # Forzamos un split
        hand.cards = [Card(9, Suit.HEARTS), Card(9, Suit.SPADES)]
        actions = {0: 'split'}
        state, reward, done = self.env.step(actions)
        # Intentamos doblar en una de las manos divididas
        actions = {0: 'double', 1: 'stand'}
        state, reward, done = self.env.step(actions)
        hand0 = self.env.player.hands[0]
        self.assertTrue(hand0.has_doubled)
        self.assertTrue(hand0.is_finished)
        # Verificar que el juego continúa si aún hay manos activas
        if self.env.player.all_hands_finished():
            self.assertTrue(done)
        else:
            self.assertFalse(done)



    def test_dealer_blackjack(self):
        self.env.reset()
        # Forzamos que el dealer tenga blackjack
        dealer_hand = self.env.dealer.hand
        dealer_hand.cards = [Card(1, Suit.CLUBS), Card(10, Suit.DIAMONDS)]  # Blackjack
        # El jugador se planta
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        player_hand = self.env.player.hands[0]
        if player_hand.is_blackjack():
            self.assertEqual(reward, 0)  # Empate
        else:
            self.assertLess(reward, 0)  # Jugador pierde

    def test_player_blackjack(self):
        self.env.reset()
        # Forzamos que el jugador tenga blackjack
        player_hand = self.env.player.hands[0]
        player_hand.cards = [Card(1, Suit.HEARTS), Card(10, Suit.SPADES)]  # Blackjack
        # El dealer no tiene blackjack
        dealer_hand = self.env.dealer.hand
        dealer_hand.cards = [Card(10, Suit.CLUBS), Card(7, Suit.DIAMONDS)]
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        self.assertGreater(reward, 0)

    def test_both_blackjack(self):
        self.env.reset()
        # Forzamos que tanto el jugador como el dealer tengan blackjack
        player_hand = self.env.player.hands[0]
        player_hand.cards = [Card(1, Suit.HEARTS), Card(10, Suit.SPADES)]
        dealer_hand = self.env.dealer.hand
        dealer_hand.cards = [Card(1, Suit.CLUBS), Card(10, Suit.DIAMONDS)]
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        self.assertEqual(reward, 0)  # Empate

    def test_insurance_not_implemented(self):
        # Dado que el seguro no está implementado, intentamos verificar que no afecta el juego
        pass  # No es necesario implementar esta prueba ya que el seguro no está en el entorno

    def test_dealer_bust(self):
        self.env.reset()
        # Forzamos que el dealer se pase
        dealer_hand = self.env.dealer.hand
        dealer_hand.cards = [Card(10, Suit.CLUBS), Card(9, Suit.DIAMONDS), Card(5, Suit.HEARTS)]  # Total 24
        actions = {0: 'stand'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        self.assertGreater(reward, 0)

    def test_player_bust(self):
        self.env.reset()
        # Forzamos que el jugador se pase
        player_hand = self.env.player.hands[0]
        player_hand.cards = [Card(10, Suit.HEARTS), Card(9, Suit.SPADES), Card(5, Suit.CLUBS)]  # Total 24
        actions = {0: 'hit'}
        state, reward, done = self.env.step(actions)
        self.assertTrue(done)
        self.assertLess(reward, 0)

if __name__ == '__main__':
    unittest.main()
