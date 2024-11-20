# environment/rewards.py

def calculate_reward(player, dealer, done):
    if not done:
        return 0
    total_reward = 0
    for hand in player.hands:
        if hand.is_bust():
            total_reward -= hand.bet_size
        elif dealer.is_bust():
            total_reward += hand.bet_size
        else:
            player_value = hand.get_value()
            dealer_value = dealer.get_value()
            if player_value > dealer_value:
                total_reward += hand.bet_size
            elif player_value < dealer_value:
                total_reward -= hand.bet_size
            else:
                pass  # Tie, no reward change
    return total_reward
