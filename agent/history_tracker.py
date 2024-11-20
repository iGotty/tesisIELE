# agent/history_tracker.py

import json

class HistoryTracker:
    def __init__(self):
        self.history = []

    def record_action(self, category, state, action):
        # Record each decision
        decision_record = {
            "state": {
                "player_value": state["player_value"],
                "dealer_upcard": state["dealer_upcard"],
                "true_count": state["true_count"],
                "hand_type": state["hand_type"]  # Include hand_type in the log
            },
            "category": category,
            "action_taken": action
        }
        self.history.append(decision_record)

    def save_history(self, filename="decision_history.json"):
        with open(filename, "w") as f:
            json.dump(self.history, f, indent=4)
    