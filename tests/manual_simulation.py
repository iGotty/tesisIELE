import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.blackjack_game import BlackjackGameEnv

def manual_simulation():
    env = BlackjackGameEnv()
    state = env.reset()
    print("Estado inicial:", state)

    actions = ["hit", "stand", "double", "split"]

    while not env.done:
        print("\nAcciones disponibles: 'hit', 'stand', 'double', 'split'")
        action = input("Elige una acción: ").strip()

        if action not in actions:
            print("Acción inválida. Por favor elige 'hit', 'stand', 'double', o 'split'.")
            continue

        state, reward, done = env.step(action)
        print(f"\nAcción tomada: {action}")
        print("Estado actual:", state)
        print("Recompensa:", reward)
        print("Juego terminado:", done)

    print("\nFin del juego.")
    print("Resultado final del jugador:", env.player.hand.get_value())
    print("Resultado final del dealer:", env.dealer.hand.get_value())

if __name__ == "__main__":
    manual_simulation()
