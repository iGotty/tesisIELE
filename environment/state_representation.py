# environment/state_representation.py

def get_state_representation(player, dealer, true_count):
    if player.hands:
        first_hand = player.hands[0]
        return get_state_representation_for_hand(first_hand, dealer, true_count)
    else:
        # Estado por defecto si no hay manos
        return {
            'player_value': 0,
            'dealer_upcard': dealer.get_upcard_value(),
            'true_count': true_count,
            'hand_type': 2  # 'hard_totals' por defecto
        }

def get_state_representation_for_hand(hand, dealer, true_count):
    # Obtener el valor de la mano del jugador
    player_value = hand.get_value()
    # Obtener el valor de la carta visible del dealer
    dealer_upcard_value = dealer.get_upcard_value()
    # Determinar el tipo de mano
    if hand.can_split():
        hand_type = 1  # 'pair_splitting'
    elif hand.is_soft():
        hand_type = 0  # 'soft_totals'
    else:
        hand_type = 2  # 'hard_totals'
    # Crear el diccionario de estado
    state = {
        'player_value': player_value,
        'dealer_upcard': dealer_upcard_value,
        'true_count': true_count,
        'hand_type': hand_type
    }
    return state
