"""
game_engine.py - Pygame Solitaire Environment with Laya Agent Interface & Visuals

Features:
- Pygame UI rendering (30~60 FPS) with crisp vector-rendered playing cards.
- Human interactive play & Laya Agent autonomous play.
- 0.3 ~ 0.5s configurable turn delay for human observation.
- State extractor summarizing context in <512 tokens (key-value format).
- Action generator producing valid options list.
- Integrated DataLogger for JSONL dataset generation.
- CLI options for headless batch data generation or interactive GUI play.
"""

import sys
import os
import random
import time
import argparse
from typing import List, Tuple, Optional, Dict, Any

# Suppress Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from data_logger import DataLogger
from laya_agent import LayaAgent


# ==========================================
# Card & Deck Representation
# ==========================================

SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
SUIT_SYMBOLS = {
    "Hearts": "♥",
    "Diamonds": "♦",
    "Clubs": "♣",
    "Spades": "♠"
}
SUIT_COLORS = {
    "Hearts": (220, 38, 38),     # Crimson Red
    "Diamonds": (220, 38, 38),   # Crimson Red
    "Clubs": (24, 24, 27),       # Dark Slate / Black
    "Spades": (24, 24, 27)       # Dark Slate / Black
}

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
RANK_VALUES = {r: i + 1 for i, r in enumerate(RANKS)}


class Card:
    """Represents a single playing card."""
    def __init__(self, suit: str, rank: str, face_up: bool = False):
        self.suit = suit
        self.rank = rank
        self.value = RANK_VALUES[rank]
        self.face_up = face_up

    @property
    def is_red(self) -> bool:
        return self.suit in ("Hearts", "Diamonds")

    def __repr__(self) -> str:
        if not self.face_up:
            return "[Hidden]"
        return f"{self.suit} {self.rank}"

    def short_str(self) -> str:
        if not self.face_up:
            return "[?]"
        sym = SUIT_SYMBOLS[self.suit]
        return f"{sym}{self.rank}"


# ==========================================
# Action Representation
# ==========================================

class Action:
    def __init__(self, action_type: str, source: str, target: str, card_info: str = ""):
        self.action_type = action_type  # 'draw', 'reset_stock', 'waste_to_foundation', etc.
        self.source = source
        self.target = target
        self.card_info = card_info

    def to_option_string(self) -> str:
        if self.action_type == "draw":
            return "Draw Stock"
        elif self.action_type == "reset_stock":
            return "Reset Waste to Stock"
        elif self.action_type == "waste_to_foundation":
            return f"Move Waste {self.card_info} to Foundation"
        elif self.action_type == "waste_to_tableau":
            return f"Move Waste {self.card_info} to {self.target}"
        elif self.action_type == "tableau_to_foundation":
            return f"Move {self.card_info} from {self.source} to Foundation"
        elif self.action_type == "tableau_to_tableau":
            return f"Move {self.card_info} from {self.source} to {self.target}"
        elif self.action_type == "reveal_card":
            return f"Reveal face-down card in {self.source}"
        return "No Action"

    def is_beneficial(self) -> bool:
        """Determines if this move represents positive progress."""
        if "foundation" in self.action_type or "reveal" in self.action_type:
            return True
        if "tableau" in self.action_type and "Empty" in self.target:
            return True
        if self.action_type == "waste_to_tableau":
            return True
        return False


class CardAnimation:
    """Tracks active card movement along a quadratic Bezier trajectory."""
    def __init__(
        self,
        cards: List[Card],
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        duration: float = 0.38,
        label: str = ""
    ):
        self.cards = cards
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.start_time = time.time()
        self.label = label

        # Quadratic Bezier Control Point (curves upwards)
        mid_x = (start_pos[0] + end_pos[0]) / 2
        min_y = min(start_pos[1], end_pos[1])
        # Curve height depends on distance
        dist = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
        arc_height = max(40.0, min(120.0, dist * 0.25))
        self.ctrl_pos = (mid_x, min_y - arc_height)

    def get_progress(self) -> float:
        elapsed = time.time() - self.start_time
        return min(1.0, max(0.0, elapsed / self.duration))

    def get_current_pos(self) -> Tuple[int, int]:
        # Ease-out cubic: t = 1 - (1-p)^3
        p = self.get_progress()
        t = 1.0 - (1.0 - p) ** 3

        p0 = self.start_pos
        p1 = self.ctrl_pos
        p2 = self.end_pos

        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        return (int(x), int(y))

    def get_trajectory_points(self, num_points: int = 30) -> List[Tuple[int, int]]:
        pts = []
        p0 = (self.start_pos[0] + 43, self.start_pos[1] + 60)
        p1 = (self.ctrl_pos[0] + 43, self.ctrl_pos[1] + 60)
        p2 = (self.end_pos[0] + 43, self.end_pos[1] + 60)
        for i in range(num_points + 1):
            t = i / num_points
            x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
            y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
            pts.append((int(x), int(y)))
        return pts



# ==========================================
# Solitaire Game State Engine
# ==========================================

class SolitaireEngine:
    """
    Standard Klondike Solitaire rules engine:
    - 7 Tableau columns
    - 4 Foundation piles (Hearts, Diamonds, Clubs, Spades)
    - Stock pile and Waste pile
    - Built-in State Hash, Loop / Cycle Detector & Action Masking
    """

    def __init__(self):
        self.stock: List[Card] = []
        self.waste: List[Card] = []
        self.foundations: Dict[str, List[Card]] = {suit: [] for suit in SUITS}
        self.tableaus: List[List[Card]] = [[] for _ in range(7)]
        self.move_count = 0
        self.moves_since_progress = 0
        self.visited_states: Dict[str, int] = {}
        self.last_executed_action: Optional[Action] = None

        self.start_time = 0.0
        self.clear_time = 0.0

        self.reset()

    def reset(self):
        """Initializes and deals a fresh game."""
        deck = [Card(suit, rank, face_up=False) for suit in SUITS for rank in RANKS]
        random.shuffle(deck)

        self.stock = []
        self.waste = []
        self.foundations = {suit: [] for suit in SUITS}
        self.tableaus = [[] for _ in range(7)]
        self.move_count = 0
        self.moves_since_progress = 0
        self.visited_states = {}
        self.last_executed_action = None

        self.start_time = time.time()
        self.clear_time = 0.0


        # Deal to 7 tableaus: col 0 has 1 card, col 1 has 2, ..., col 6 has 7
        for col in range(7):
            for row in range(col + 1):
                card = deck.pop()
                if row == col:
                    card.face_up = True
                self.tableaus[col].append(card)

        # Remaining cards go to Stock
        self.stock = deck
        for c in self.stock:
            c.face_up = False

        self._record_current_state()

    def get_state_hash(self) -> str:
        """Generates a canonical hash string of current table layout to detect cycles/loops."""
        found_state = tuple(f[-1].short_str() if f else "_" for f in self.foundations.values())
        tab_state = tuple(
            (len(col), col[-1].short_str() if col and col[-1].face_up else "_", sum(1 for c in col if not c.face_up))
            for col in self.tableaus
        )
        waste_top = self.waste[-1].short_str() if self.waste else "_"
        stock_len = len(self.stock)
        return f"F:{found_state}|T:{tab_state}|W:{waste_top}|S:{stock_len}"

    def _record_current_state(self):
        h = self.get_state_hash()
        self.visited_states[h] = self.visited_states.get(h, 0) + 1

    def is_loop_detected(self) -> bool:
        """Returns True if the current exact game state has recurred 2 or more times."""
        h = self.get_state_hash()
        return self.visited_states.get(h, 0) >= 2

    def is_stalled(self, max_stalled_moves: int = 25) -> bool:
        """Returns True if no card has progressed to foundation or revealed for N moves."""
        return self.moves_since_progress >= max_stalled_moves

    def get_total_foundation_cards(self) -> int:
        return sum(len(f) for f in self.foundations.values())

    def is_won(self) -> bool:
        """Game is won if all 4 foundations have 13 cards (Kings)."""
        won = all(len(f) == 13 for f in self.foundations.values())
        if won and self.clear_time == 0.0:
            self.clear_time = round(time.time() - self.start_time, 2)
        return won

    # ----------------------------------------------------
    # State Extractor (< 512 tokens concise representation)
    # ----------------------------------------------------
    def get_context_summary(self) -> str:
        """
        Summarizes current game state into a concise text representation.
        Designed to strictly fit well within 512 tokens.
        """
        lines = []
        lines.append(f"Stock Remaining: {len(self.stock)}")
        
        waste_str = repr(self.waste[-1]) if self.waste else "None"
        lines.append(f"Waste Top: {waste_str}")

        found_items = []
        for suit in SUITS:
            pile = self.foundations[suit]
            if pile:
                found_items.append(f"{suit} {pile[-1].rank}")
        lines.append(f"Foundations: [{', '.join(found_items)}]")

        tableau_items = []
        for idx, tab in enumerate(self.tableaus):
            if not tab:
                tableau_items.append(f"Col {idx+1}: Empty")
            else:
                top = tab[-1]
                hidden = sum(1 for c in tab if not c.face_up)
                tableau_items.append(f"Col {idx+1}: {repr(top)} (Hidden: {hidden})")
        lines.append("Tableaus: [" + "; ".join(tableau_items) + "]")

        return "\n".join(lines)

    # ----------------------------------------------------
    # Action Generator
    # ----------------------------------------------------
    def get_valid_actions(self) -> List[Action]:
        """Generates all legal actions available in the current state."""
        actions: List[Action] = []

        # 1. Reveal face-down card if top of any tableau is face-down
        for idx, tab in enumerate(self.tableaus):
            if tab and not tab[-1].face_up:
                actions.append(Action("reveal_card", f"Tableau Col {idx+1}", ""))

        # 2. Waste to Foundation
        if self.waste:
            top_waste = self.waste[-1]
            found = self.foundations[top_waste.suit]
            target_val = 1 if not found else found[-1].value + 1
            if top_waste.value == target_val:
                actions.append(Action("waste_to_foundation", "Waste", f"Foundation {top_waste.suit}", repr(top_waste)))

        # 3. Waste to Tableau
        if self.waste:
            top_waste = self.waste[-1]
            for idx, tab in enumerate(self.tableaus):
                if not tab:
                    if top_waste.value == 13:  # Only Kings can be placed on empty tableau
                        actions.append(Action("waste_to_tableau", "Waste", f"Empty Tableau Col {idx+1}", repr(top_waste)))
                else:
                    target_card = tab[-1]
                    if target_card.face_up and (target_card.is_red != top_waste.is_red) and (target_card.value == top_waste.value + 1):
                        actions.append(Action("waste_to_tableau", "Waste", f"Tableau Col {idx+1} ({repr(target_card)})", repr(top_waste)))

        # 4. Tableau to Foundation
        for idx, tab in enumerate(self.tableaus):
            if tab and tab[-1].face_up:
                top_card = tab[-1]
                found = self.foundations[top_card.suit]
                target_val = 1 if not found else found[-1].value + 1
                if top_card.value == target_val:
                    actions.append(Action("tableau_to_foundation", f"Tableau Col {idx+1}", f"Foundation {top_card.suit}", repr(top_card)))

        # 5. Tableau to Tableau
        for src_idx, src_tab in enumerate(self.tableaus):
            # Find face-up cards that can be moved
            face_up_indices = [i for i, c in enumerate(src_tab) if c.face_up]
            if not face_up_indices:
                continue

            for card_pos in face_up_indices:
                moving_card = src_tab[card_pos]
                # If moving the bottom-most face-up card and there are no face-down cards below it,
                # moving it to an empty column is redundant
                is_base_card = (card_pos == 0)

                for dst_idx, dst_tab in enumerate(self.tableaus):
                    if src_idx == dst_idx:
                        continue
                    if not dst_tab:
                        # Moving a King to empty column
                        if moving_card.value == 13 and not is_base_card:
                            actions.append(Action(
                                "tableau_to_tableau",
                                f"Tableau Col {src_idx+1}",
                                f"Empty Tableau Col {dst_idx+1}",
                                repr(moving_card)
                            ))
                    else:
                        dst_card = dst_tab[-1]
                        if dst_card.face_up and (dst_card.is_red != moving_card.is_red) and (dst_card.value == moving_card.value + 1):
                            # Action masking: prevent immediately reversing the move from previous turn
                            if self.last_executed_action and self.last_executed_action.action_type == "tableau_to_tableau":
                                if self.last_executed_action.card_info == repr(moving_card):
                                    if f"Col {dst_idx+1}" in self.last_executed_action.source:
                                        continue

                            actions.append(Action(
                                "tableau_to_tableau",
                                f"Tableau Col {src_idx+1}",
                                f"Tableau Col {dst_idx+1} ({repr(dst_card)})",
                                repr(moving_card)
                            ))

        # 6. Draw Stock / Reset Waste
        if self.stock:
            actions.append(Action("draw", "Stock", "Waste"))
        elif self.waste:
            actions.append(Action("reset_stock", "Waste", "Stock"))

        if not actions:
            actions.append(Action("none", "", ""))

        return actions

    def get_valid_option_strings(self) -> List[str]:
        """Returns option strings directly."""
        actions = self.get_valid_actions()
        return [act.to_option_string() for act in actions]

    def execute_action(self, action: Action) -> bool:
        """Applies an Action object directly to game state with loop and progress tracking."""
        self.move_count += 1
        executed = False
        is_progress = False

        if action.action_type == "draw":
            if self.stock:
                c = self.stock.pop()
                c.face_up = True
                self.waste.append(c)
                executed = True

        elif action.action_type == "reset_stock":
            if not self.stock and self.waste:
                self.stock = list(reversed(self.waste))
                for c in self.stock:
                    c.face_up = False
                self.waste.clear()
                executed = True

        elif action.action_type == "reveal_card":
            col = int(action.source.split()[-1]) - 1
            if self.tableaus[col] and not self.tableaus[col][-1].face_up:
                self.tableaus[col][-1].face_up = True
                executed = True
                is_progress = True

        elif action.action_type == "waste_to_foundation":
            if self.waste:
                c = self.waste.pop()
                self.foundations[c.suit].append(c)
                executed = True
                is_progress = True

        elif action.action_type == "waste_to_tableau":
            col = int(action.target.split("Col ")[-1].split()[0].split("(")[0]) - 1
            if self.waste:
                c = self.waste.pop()
                self.tableaus[col].append(c)
                executed = True
                is_progress = True

        elif action.action_type == "tableau_to_foundation":
            src_col = int(action.source.split()[-1]) - 1
            if self.tableaus[src_col] and self.tableaus[src_col][-1].face_up:
                c = self.tableaus[src_col].pop()
                self.foundations[c.suit].append(c)
                if self.tableaus[src_col] and not self.tableaus[src_col][-1].face_up:
                    self.tableaus[src_col][-1].face_up = True
                executed = True
                is_progress = True

        elif action.action_type == "tableau_to_tableau":
            src_col = int(action.source.split()[-1]) - 1
            dst_col = int(action.target.split("Col ")[-1].split()[0].split("(")[0]) - 1
            src_tab = self.tableaus[src_col]
            move_idx = -1
            for i, c in enumerate(src_tab):
                if c.face_up and repr(c) == action.card_info:
                    move_idx = i
                    break
            if move_idx == -1 and src_tab:
                move_idx = len(src_tab) - 1

            moving_substack = src_tab[move_idx:]
            self.tableaus[src_col] = src_tab[:move_idx]
            self.tableaus[dst_col].extend(moving_substack)

            if self.tableaus[src_col] and not self.tableaus[src_col][-1].face_up:
                self.tableaus[src_col][-1].face_up = True
                is_progress = True
            executed = True

        if executed:
            if is_progress:
                self.moves_since_progress = 0
            else:
                self.moves_since_progress += 1
            self.last_executed_action = action
            self._record_current_state()
            return True

        return False

    def execute_option_string(self, option_str: str) -> bool:
        """Parses an option string, matches with valid actions, and executes."""
        actions = self.get_valid_actions()
        for act in actions:
            if act.to_option_string() == option_str:
                return self.execute_action(act)
        # If no exact match found, pick first valid action
        if actions and actions[0].action_type != "none":
            return self.execute_action(actions[0])
        return False


# ==========================================
# Pygame GUI Renderer & Game Loop
# ==========================================

class SolitaireGUI:
    """
    Renders Solitaire and manages Human / Agent game loop.
    Includes real-time Laya Decision Log overlay with full multi-line text wrapping.
    """

    WIDTH = 1440
    HEIGHT = 760
    BOARD_WIDTH = 920
    CARD_W = 86
    CARD_H = 120
    CARD_CORNER = 8
    TABLEAU_OFFSET_Y = 24

    # Colors
    BG_COLOR = (24, 76, 48)         # Classic Felt Green
    TOP_BAR_COLOR = (18, 54, 34)    # Dark Green Top Panel
    CARD_BG = (255, 255, 255)       # Card White
    CARD_BORDER = (180, 180, 180)   # Subtle Card Border
    CARD_BACK = (45, 90, 160)       # Blue Backing
    TEXT_COLOR = (240, 240, 240)
    ACCENT_COLOR = (245, 158, 11)   # Gold / Amber
    OVERLAY_BG = (15, 23, 42)       # Slate 900
    OVERLAY_CARD = (30, 41, 59)     # Slate 800

    def __init__(
        self,
        bot_mode: bool = True,
        bot_delay: float = 0.4,
        enable_logging: bool = True,
        use_heuristic: bool = True,
        model_path: str = "convaiinnovations/laya"
    ):
        pygame.init()
        pygame.display.set_caption("Laya Solitaire AI - Decision Monitor")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 20, bold=True)
        self.font_large = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 24, bold=True)
        self.font_mid = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15, bold=True)
        self.font_card = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13)
        self.font_tiny = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 11)

        # Core Components
        self.engine = SolitaireEngine()
        self.logger = DataLogger(filter_beneficial=False) if enable_logging else None
        self.agent = LayaAgent(model_name_or_path=model_path, use_heuristic=use_heuristic)

        if self.logger:
            self.logger.start_episode()

        # Runtime State
        self.bot_mode = bot_mode
        self.bot_delay = bot_delay
        self.enable_logging = enable_logging
        self.last_action_time = time.time()
        self.last_action_text = "Initialized"
        self.running = True
        self.show_overlay = True

        # Animation & Trajectory Path State
        self.current_animation: Optional[CardAnimation] = None
        self.pending_action: Optional[Action] = None
        self.pending_log_data: Optional[Tuple[str, List[str], str, bool]] = None
        self.last_trajectory_points: List[Tuple[int, int]] = []
        self.last_trajectory_label: str = ""
        self.hidden_cards_in_flight: List[Card] = []

        # Laya Decision Monitor Telemetry
        self.decision_telemetry: Dict[str, Any] = {
            "context": "Waiting for first step...",
            "options": [],
            "target": "None",
            "is_beneficial": False,
            "turn": 0,
            "history": []
        }

    def _compute_action_endpoints(self, action: Action) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], List[Card]]:
        """Calculates start and target coordinates for animated card transitions."""
        stock_pos = (40, 80)
        waste_pos = (40 + self.CARD_W + 20, 80)
        found_start_x = self.BOARD_WIDTH - (4 * (self.CARD_W + 16)) - 40
        col_gap = (self.BOARD_WIDTH - 80 - (7 * self.CARD_W)) // 6

        if action.action_type == "draw":
            if self.engine.stock:
                top_stock = self.engine.stock[-1]
                card = Card(top_stock.suit, top_stock.rank, face_up=True)
                return stock_pos, waste_pos, [card]

        elif action.action_type == "reset_stock":
            if self.engine.waste:
                top_waste = self.engine.waste[-1]
                card = Card(top_waste.suit, top_waste.rank, face_up=False)
                return waste_pos, stock_pos, [card]

        elif action.action_type == "waste_to_foundation":
            if self.engine.waste:
                card = self.engine.waste[-1]
                suit_idx = SUITS.index(card.suit)
                fx = found_start_x + suit_idx * (self.CARD_W + 16)
                return waste_pos, (fx, 80), [card]

        elif action.action_type == "waste_to_tableau":
            if self.engine.waste:
                card = self.engine.waste[-1]
                col = int(action.target.split("Col ")[-1].split()[0].split("(")[0]) - 1
                tx = 40 + col * (self.CARD_W + col_gap)
                ty = 230 + len(self.engine.tableaus[col]) * self.TABLEAU_OFFSET_Y
                return waste_pos, (tx, ty), [card]

        elif action.action_type == "tableau_to_foundation":
            src_col = int(action.source.split()[-1]) - 1
            if self.engine.tableaus[src_col] and self.engine.tableaus[src_col][-1].face_up:
                card = self.engine.tableaus[src_col][-1]
                suit_idx = SUITS.index(card.suit)
                fx = found_start_x + suit_idx * (self.CARD_W + 16)
                sx = 40 + src_col * (self.CARD_W + col_gap)
                sy = 230 + max(0, len(self.engine.tableaus[src_col]) - 1) * self.TABLEAU_OFFSET_Y
                return (sx, sy), (fx, 80), [card]

        elif action.action_type == "tableau_to_tableau":
            src_col = int(action.source.split()[-1]) - 1
            dst_col = int(action.target.split("Col ")[-1].split()[0].split("(")[0]) - 1
            src_tab = self.engine.tableaus[src_col]
            move_idx = -1
            for i, c in enumerate(src_tab):
                if c.face_up and repr(c) == action.card_info:
                    move_idx = i
                    break
            if move_idx == -1 and src_tab:
                move_idx = len(src_tab) - 1

            moving_cards = src_tab[move_idx:]
            sx = 40 + src_col * (self.CARD_W + col_gap)
            sy = 230 + move_idx * self.TABLEAU_OFFSET_Y
            dx = 40 + dst_col * (self.CARD_W + col_gap)
            dy = 230 + len(self.engine.tableaus[dst_col]) * self.TABLEAU_OFFSET_Y
            return (sx, sy), (dx, dy), moving_cards

        return None, None, []

    def run(self):
        """Main game loop running at 60 FPS."""
        while self.running:
            self._handle_events()

            # Bot step if autonomous mode is on and not currently animating
            current_time = time.time()
            if self.bot_mode and (self.current_animation is None) and (current_time - self.last_action_time >= self.bot_delay):
                self._bot_step()
                self.last_action_time = current_time

            self._draw()
            self.clock.tick(60)

        pygame.quit()

    def _bot_step(self):
        """Executes a single autonomous step with loop detection, stall prevention, and episode filtering."""
        if self.current_animation is not None:
            return

        # 1. Check Win
        if self.engine.is_won():
            if self.logger:
                committed, count = self.logger.commit_episode(won=True)
                self.last_action_text = f"🎉 Won! Saved {count} training steps."
            else:
                self.last_action_text = "🎉 Game Won!"
            self.bot_mode = False
            return

        # 2. Check Loop Detection (Cycle Breaker)
        if self.engine.is_loop_detected():
            if self.logger:
                _, dropped = self.logger.discard_episode(reason="loop_detected")
                self.last_action_text = f"⚠️ Loop Detected! Discarded {dropped} steps & Auto-Reset."
            else:
                self.last_action_text = "⚠️ Loop Detected! Auto-Resetting."
            self.engine.reset()
            if self.logger:
                self.logger.start_episode()
            return

        # 3. Check Stall (No progress for 25 consecutive moves)
        if self.engine.is_stalled(25):
            if self.logger:
                _, dropped = self.logger.discard_episode(reason="stall_detected")
                self.last_action_text = f"⏳ Stalled! Discarded {dropped} steps & Auto-Reset."
            else:
                self.last_action_text = "⏳ Stalled! Auto-Resetting."
            self.engine.reset()
            if self.logger:
                self.logger.start_episode()
            return

        # 4. Check Max Moves (120 moves limit)
        if self.engine.move_count >= 120:
            if self.logger:
                committed, count = self.logger.commit_episode(
                    foundation_progress=self.engine.get_total_foundation_cards(),
                    min_progress_required=4
                )
                if committed:
                    self.last_action_text = f"Max Moves: Saved {count} quality steps & Resetting."
                else:
                    self.last_action_text = f"Max Moves: Discarded low-progress episode & Resetting."
            else:
                self.last_action_text = "Max Moves reached. Resetting."
            self.engine.reset()
            if self.logger:
                self.logger.start_episode()
            return

        context = self.engine.get_context_summary()
        valid_actions = self.engine.get_valid_actions()
        options = [act.to_option_string() for act in valid_actions]

        if not options or (len(options) == 1 and options[0] == "No Action"):
            if self.logger:
                self.logger.discard_episode(reason="no_moves")
            self.last_action_text = "No valid moves remaining. Auto-Resetting."
            self.engine.reset()
            if self.logger:
                self.logger.start_episode()
            return

        best_option, is_beneficial = self.agent.select_action(context, options)

        # Match action object
        matched_act = None
        for act in valid_actions:
            if act.to_option_string() == best_option:
                matched_act = act
                break
        if matched_act is None and valid_actions:
            matched_act = valid_actions[0]
            best_option = matched_act.to_option_string()

        start_pos, end_pos, moving_cards = self._compute_action_endpoints(matched_act)

        # Update Decision Telemetry for Overlay
        self.decision_telemetry["context"] = context
        self.decision_telemetry["options"] = list(options)
        self.decision_telemetry["target"] = best_option
        self.decision_telemetry["is_beneficial"] = is_beneficial
        self.decision_telemetry["turn"] = self.engine.move_count + 1
        history = self.decision_telemetry.setdefault("history", [])
        history.append({
            "turn": self.decision_telemetry["turn"],
            "target": best_option,
            "is_beneficial": is_beneficial
        })
        if len(history) > 8:
            history.pop(0)

        if start_pos and end_pos and moving_cards:
            # Launch smooth trajectory animation
            self.current_animation = CardAnimation(
                cards=moving_cards,
                start_pos=start_pos,
                end_pos=end_pos,
                duration=0.36,
                label=best_option
            )
            self.pending_action = matched_act
            self.pending_log_data = (context, options, best_option, is_beneficial)
            self.last_trajectory_points = self.current_animation.get_trajectory_points()
            self.last_trajectory_label = best_option
            self.hidden_cards_in_flight = list(moving_cards)
            self.last_action_text = f"Bot: {best_option}"
        else:
            # Immediate action (e.g. reveal in place)
            executed = self.engine.execute_action(matched_act)
            if executed:
                self.last_action_text = f"Bot: {best_option}"
                if self.logger:
                    self.logger.log_step(context, options, best_option, is_beneficial)
            else:
                self.last_action_text = f"Action failed: {best_option}"

    def _handle_events(self):
        """Processes user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.bot_mode = not self.bot_mode
                    self.last_action_text = f"Bot Mode: {'ENABLED' if self.bot_mode else 'PAUSED'}"
                elif event.key == pygame.K_n:
                    # Single step bot
                    self._bot_step()
                elif event.key == pygame.K_r:
                    if self.logger:
                        self.logger.discard_episode(reason="manual_reset")
                        self.logger.start_episode()
                    self.engine.reset()
                    self.last_action_text = "Game Reset."
                elif event.key == pygame.K_l:
                    self.enable_logging = not self.enable_logging
                    self.last_action_text = f"Logging: {'ON' if self.enable_logging else 'OFF'}"
                elif event.key == pygame.K_o:
                    self.show_overlay = not self.show_overlay
                    self.last_action_text = f"Laya Overlay: {'ON' if self.show_overlay else 'OFF'}"

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Human quick interaction: click Stock to draw
                mouse_pos = pygame.mouse.get_pos()
                stock_rect = pygame.Rect(40, 80, self.CARD_W, self.CARD_H)
                if stock_rect.collidepoint(mouse_pos):
                    if self.engine.stock:
                        self.engine.execute_option_string("Draw Stock")
                        self.last_action_text = "Human: Draw Stock"
                    elif self.engine.waste:
                        self.engine.execute_option_string("Reset Waste to Stock")
                        self.last_action_text = "Human: Reset Waste"

    # ----------------------------------------------------
    # Rendering Helper Functions
    # ----------------------------------------------------
    def _draw_card(self, surface: pygame.Surface, card: Card, x: int, y: int):
        rect = pygame.Rect(x, y, self.CARD_W, self.CARD_H)

        if not card.face_up:
            # Face-down card design
            pygame.draw.rect(surface, (220, 220, 230), rect, border_radius=self.CARD_CORNER)
            inner_rect = rect.inflate(-6, -6)
            pygame.draw.rect(surface, self.CARD_BACK, inner_rect, border_radius=self.CARD_CORNER - 2)
            pygame.draw.rect(surface, (70, 130, 210), inner_rect.inflate(-8, -8), 2, border_radius=self.CARD_CORNER - 4)
            return

        # Face-up card
        pygame.draw.rect(surface, self.CARD_BG, rect, border_radius=self.CARD_CORNER)
        pygame.draw.rect(surface, self.CARD_BORDER, rect, 1, border_radius=self.CARD_CORNER)

        color = SUIT_COLORS[card.suit]
        symbol = SUIT_SYMBOLS[card.suit]

        # Top-left corner
        rank_surf = self.font_card.render(card.rank, True, color)
        suit_surf = self.font_card.render(symbol, True, color)
        surface.blit(rank_surf, (x + 6, y + 4))
        surface.blit(suit_surf, (x + 6, y + 22))

        # Center large suit symbol
        center_sym_surf = self.font_large.render(symbol, True, color)
        sym_rect = center_sym_surf.get_rect(center=(x + self.CARD_W // 2, y + self.CARD_H // 2))
        surface.blit(center_sym_surf, sym_rect)

        # Bottom-right inverted corner
        surface.blit(rank_surf, (x + self.CARD_W - rank_surf.get_width() - 6, y + self.CARD_H - rank_surf.get_height() - 22))
        surface.blit(suit_surf, (x + self.CARD_W - suit_surf.get_width() - 6, y + self.CARD_H - suit_surf.get_height() - 4))

    def _draw_empty_slot(self, surface: pygame.Surface, x: int, y: int, label: str = ""):
        rect = pygame.Rect(x, y, self.CARD_W, self.CARD_H)
        pygame.draw.rect(surface, (20, 60, 38), rect, border_radius=self.CARD_CORNER)
        pygame.draw.rect(surface, (50, 110, 75), rect, 2, border_radius=self.CARD_CORNER)
        if label:
            txt = self.font_small.render(label, True, (80, 140, 105))
            txt_rect = txt.get_rect(center=rect.center)
            surface.blit(txt, txt_rect)

    def _render_wrapped_text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        color: Tuple[int, int, int],
        x: int,
        y: int,
        max_width: int,
        line_spacing: int = 2,
        max_lines: int = 4
    ) -> int:
        """Renders multi-line text wrapped within max_width. Returns bottom y reached."""
        words = text.split(" ")
        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            test_w = font.size(test_line)[0]
            if test_w <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        cur_y = y
        line_h = font.get_linesize()
        for idx, line in enumerate(lines[:max_lines]):
            if idx == max_lines - 1 and len(lines) > max_lines:
                line = line + "..."
            line_surf = font.render(line, True, color)
            surface.blit(line_surf, (x, cur_y))
            cur_y += line_h + line_spacing

        return cur_y

    def _draw_overlay(self):
        """Renders real-time Laya Decision Log overlay on the right panel with full multi-line text wrapping."""
        if not self.show_overlay:
            # Draw minimal toggle prompt tab
            tab_rect = pygame.Rect(self.WIDTH - 110, 14, 100, 28)
            pygame.draw.rect(self.screen, self.OVERLAY_CARD, tab_rect, border_radius=6)
            pygame.draw.rect(self.screen, (51, 65, 85), tab_rect, 1, border_radius=6)
            tab_txt = self.font_small.render("[O] Show Log", True, (148, 163, 184))
            self.screen.blit(tab_txt, (tab_rect.x + 12, tab_rect.y + 6))
            return

        panel_x = self.BOARD_WIDTH
        panel_w = self.WIDTH - self.BOARD_WIDTH
        panel_h = self.HEIGHT
        card_w = panel_w - 24
        inner_text_w = card_w - 24

        # 1. Panel Background
        panel_rect = pygame.Rect(panel_x, 0, panel_w, panel_h)
        pygame.draw.rect(self.screen, self.OVERLAY_BG, panel_rect)
        pygame.draw.line(self.screen, (51, 65, 85), (panel_x, 0), (panel_x, panel_h), 2)

        # 2. Header
        hdr_surf = self.font_title.render("🧠 LAYA DECISION MONITOR", True, (241, 245, 249))
        self.screen.blit(hdr_surf, (panel_x + 16, 14))

        sub_txt = f"Model: {'Heuristic Rule Engine' if self.agent.use_heuristic else self.agent.model_name_or_path}"
        sub_surf = self.font_tiny.render(sub_txt, True, (148, 163, 184))
        self.screen.blit(sub_surf, (panel_x + 18, 38))

        # Status Badge
        badge_rect = pygame.Rect(panel_x + panel_w - 95, 14, 80, 22)
        pygame.draw.rect(self.screen, (16, 185, 129), badge_rect, border_radius=11)
        st_surf = self.font_tiny.render("● ACTIVE", True, (255, 255, 255))
        self.screen.blit(st_surf, (badge_rect.x + 14, badge_rect.y + 4))

        curr_y = 62

        # 3. Target Action Card (Dynamic Multi-line wrap)
        tgt_text = self.decision_telemetry.get("target", "No Action")
        card1_top_y = curr_y
        card1_inner_y = card1_top_y + 8

        # We first pre-draw background box with estimated height, then update
        card1_h = 78
        card1_rect = pygame.Rect(panel_x + 12, card1_top_y, card_w, card1_h)
        pygame.draw.rect(self.screen, self.OVERLAY_CARD, card1_rect, border_radius=8)
        pygame.draw.rect(self.screen, (34, 197, 94), card1_rect, 1, border_radius=8)

        lbl1 = self.font_tiny.render("TARGET ACTION (BEST DECISION)", True, (148, 163, 184))
        self.screen.blit(lbl1, (card1_rect.x + 12, card1_inner_y))

        # Render Target text with wrapping
        end_tgt_y = self._render_wrapped_text(
            surface=self.screen,
            text=tgt_text,
            font=self.font_mid,
            color=(74, 222, 128),
            x=card1_rect.x + 12,
            y=card1_inner_y + 16,
            max_width=inner_text_w,
            max_lines=2
        )

        is_ben = self.decision_telemetry.get("is_beneficial", False)
        ben_str = "Progress Move (Beneficial)" if is_ben else "Routine Move / Exploration"
        ben_color = (34, 197, 94) if is_ben else (245, 158, 11)
        ben_surf = self.font_tiny.render(f"Turn #{self.decision_telemetry.get('turn', 0)}  |  {ben_str}", True, ben_color)
        self.screen.blit(ben_surf, (card1_rect.x + 12, end_tgt_y + 2))

        curr_y += card1_h + 8

        # 3.5 Quality Guardrails & Loop Monitor Card
        card_gr_h = 56
        card_gr_rect = pygame.Rect(panel_x + 12, curr_y, card_w, card_gr_h)
        pygame.draw.rect(self.screen, self.OVERLAY_CARD, card_gr_rect, border_radius=8)

        is_loop = self.engine.is_loop_detected()
        loop_status = "⚠️ LOOP DETECTED" if is_loop else "NORMAL (0 Cycles)"
        loop_color = (239, 68, 68) if is_loop else (34, 197, 94)

        lbl_gr = self.font_tiny.render("QUALITY GUARDRAILS & LOOP MONITOR", True, (148, 163, 184))
        self.screen.blit(lbl_gr, (card_gr_rect.x + 12, card_gr_rect.y + 6))

        val_gr = self.font_tiny.render(f"Cycle Guard: {loop_status} | Stall Progress: {self.engine.moves_since_progress}/25", True, loop_color)
        self.screen.blit(val_gr, (card_gr_rect.x + 12, card_gr_rect.y + 22))

        stat_comm = self.logger.stats["episodes_committed"] if self.logger else 0
        stat_disc = self.logger.stats["episodes_discarded"] if self.logger else 0
        val_stat = self.font_tiny.render(f"Dataset Episodes -> Saved: {stat_comm}  |  Discarded (Loops): {stat_disc}", True, (245, 158, 11))
        self.screen.blit(val_stat, (card_gr_rect.x + 12, card_gr_rect.y + 38))

        curr_y += card_gr_h + 8

        # 4. Evaluated Options Section (Multi-line Wrap for Candidates)
        options = self.decision_telemetry.get("options", [])
        card2_h = 168
        card2_rect = pygame.Rect(panel_x + 12, curr_y, card_w, card2_h)
        pygame.draw.rect(self.screen, self.OVERLAY_CARD, card2_rect, border_radius=8)
        pygame.draw.rect(self.screen, (51, 65, 85), card2_rect, 1, border_radius=8)

        lbl2 = self.font_tiny.render(f"EVALUATED OPTIONS ({len(options)} Candidates)", True, (148, 163, 184))
        self.screen.blit(lbl2, (card2_rect.x + 12, card2_rect.y + 8))

        opt_y = card2_rect.y + 26
        # Render top 4-5 options with full wrapping
        for i, opt in enumerate(options[:5]):
            if opt_y >= card2_rect.y + card2_h - 18:
                break
            is_chosen = (opt == tgt_text)
            prefix = "▶ [✓] " if is_chosen else "   [  ] "
            opt_color = (187, 247, 208) if is_chosen else (203, 213, 225)
            full_opt_text = prefix + opt

            opt_y = self._render_wrapped_text(
                surface=self.screen,
                text=full_opt_text,
                font=self.font_tiny,
                color=opt_color,
                x=card2_rect.x + 12,
                y=opt_y,
                max_width=inner_text_w,
                max_lines=2
            )

        if len(options) > 5:
            more_surf = self.font_tiny.render(f"... +{len(options) - 5} more valid options", True, (100, 116, 139))
            self.screen.blit(more_surf, (card2_rect.x + 12, card2_rect.y + card2_h - 16))

        curr_y += card2_h + 8

        # 5. State Representation (Context with clean line breaks & wrapping)
        ctx = self.decision_telemetry.get("context", "")
        card3_h = 180
        card3_rect = pygame.Rect(panel_x + 12, curr_y, card_w, card3_h)
        pygame.draw.rect(self.screen, self.OVERLAY_CARD, card3_rect, border_radius=8)
        pygame.draw.rect(self.screen, (51, 65, 85), card3_rect, 1, border_radius=8)

        lbl3 = self.font_tiny.render("STATE CONTEXT (Laya Model Prompt Input)", True, (148, 163, 184))
        self.screen.blit(lbl3, (card3_rect.x + 12, card3_rect.y + 8))

        ctx_y = card3_rect.y + 26
        for line in ctx.split("\n"):
            if ctx_y >= card3_rect.y + card3_h - 14:
                break
            # Highlight key-values
            c_color = (186, 230, 253)
            ctx_y = self._render_wrapped_text(
                surface=self.screen,
                text=line,
                font=self.font_tiny,
                color=c_color,
                x=card3_rect.x + 12,
                y=ctx_y,
                max_width=inner_text_w,
                max_lines=3
            )

        curr_y += card3_h + 8

        # 6. Recent Decision History (Multi-line Wrap)
        card4_h = panel_h - curr_y - 12
        if card4_h >= 60:
            card4_rect = pygame.Rect(panel_x + 12, curr_y, card_w, card4_h)
            pygame.draw.rect(self.screen, self.OVERLAY_CARD, card4_rect, border_radius=8)
            pygame.draw.rect(self.screen, (51, 65, 85), card4_rect, 1, border_radius=8)

            lbl4 = self.font_tiny.render("RECENT DECISIONS HISTORY", True, (148, 163, 184))
            self.screen.blit(lbl4, (card4_rect.x + 12, card4_rect.y + 8))

            hist = self.decision_telemetry.get("history", [])
            hist_y = card4_rect.y + 26
            for h in reversed(hist[-4:]):
                if hist_y >= card4_rect.y + card4_h - 14:
                    break
                h_turn = h["turn"]
                h_tgt = h["target"]
                h_text = f"#{h_turn}: {h_tgt}"
                h_color = (226, 232, 240) if h.get("is_beneficial") else (148, 163, 184)
                hist_y = self._render_wrapped_text(
                    surface=self.screen,
                    text=h_text,
                    font=self.font_tiny,
                    color=h_color,
                    x=card4_rect.x + 12,
                    y=hist_y,
                    max_width=inner_text_w,
                    max_lines=2
                )

    def _draw(self):
        self.screen.fill(self.BG_COLOR)

        # Process animation completion if elapsed
        if self.current_animation:
            p = self.current_animation.get_progress()
            if p >= 1.0:
                self.engine.execute_action(self.pending_action)
                if self.logger and self.pending_log_data:
                    self.logger.log_step(*self.pending_log_data)
                self.current_animation = None
                self.pending_action = None
                self.pending_log_data = None
                self.hidden_cards_in_flight = []
                self.last_action_time = time.time()

        # 1. Top HUD / Dashboard
        hud_w = self.BOARD_WIDTH if self.show_overlay else self.WIDTH
        hud_rect = pygame.Rect(0, 0, hud_w, 56)
        pygame.draw.rect(self.screen, self.TOP_BAR_COLOR, hud_rect)
        pygame.draw.line(self.screen, (30, 80, 50), (0, 56), (hud_w, 56), 2)

        mode_text = "[BOT: AUTO]" if self.bot_mode else "[HUMAN / PAUSE]"
        mode_color = (34, 197, 94) if self.bot_mode else (234, 179, 8)
        mode_surf = self.font_mid.render(f"Mode: {mode_text}", True, mode_color)
        self.screen.blit(mode_surf, (20, 16))

        moves_surf = self.font_mid.render(f"Moves: {self.engine.move_count}", True, self.TEXT_COLOR)
        self.screen.blit(moves_surf, (220, 16))

        log_count = self.logger.count_records() if self.logger else 0
        log_surf = self.font_mid.render(f"Dataset: {log_count}", True, self.ACCENT_COLOR)
        self.screen.blit(log_surf, (330, 16))

        info_surf = self.font_small.render("[SPACE] Bot | [N] Step | [O] Overlay | [R] Reset", True, (160, 190, 170))
        self.screen.blit(info_surf, (470, 20))

        # 2. Stock and Waste (Top-Left)
        stock_x, stock_y = 40, 80
        if self.engine.stock:
            show_stock_top = True
            if self.current_animation and self.pending_action and self.pending_action.action_type == "draw":
                if len(self.engine.stock) <= 1:
                    show_stock_top = False
            if show_stock_top:
                self._draw_card(self.screen, self.engine.stock[-1], stock_x, stock_y)
            else:
                self._draw_empty_slot(self.screen, stock_x, stock_y, "Stock")
            cnt = self.font_small.render(str(len(self.engine.stock)), True, (255, 255, 255))
            self.screen.blit(cnt, (stock_x + 6, stock_y + 6))
        else:
            self._draw_empty_slot(self.screen, stock_x, stock_y, "Stock")

        waste_x = stock_x + self.CARD_W + 20
        waste_cards = [c for c in self.engine.waste if c not in self.hidden_cards_in_flight]
        if waste_cards:
            self._draw_card(self.screen, waste_cards[-1], waste_x, stock_y)
        else:
            self._draw_empty_slot(self.screen, waste_x, stock_y, "Waste")

        # 3. Foundations (Top-Right)
        found_start_x = self.BOARD_WIDTH - (4 * (self.CARD_W + 16)) - 40
        for i, suit in enumerate(SUITS):
            fx = found_start_x + i * (self.CARD_W + 16)
            pile = self.engine.foundations[suit]
            if pile:
                self._draw_card(self.screen, pile[-1], fx, stock_y)
            else:
                sym = SUIT_SYMBOLS[suit]
                self._draw_empty_slot(self.screen, fx, stock_y, sym)

        # 4. Tableaus (Bottom 7 Columns)
        tab_start_x = 40
        tab_y = 230
        col_gap = (self.BOARD_WIDTH - 80 - (7 * self.CARD_W)) // 6

        for col_idx, tab in enumerate(self.tableaus):
            tx = tab_start_x + col_idx * (self.CARD_W + col_gap)
            visible_tab = [c for c in tab if c not in self.hidden_cards_in_flight]
            if not visible_tab:
                self._draw_empty_slot(self.screen, tx, tab_y, f"Col {col_idx+1}")
            else:
                for row_idx, card in enumerate(visible_tab):
                    cy = tab_y + row_idx * self.TABLEAU_OFFSET_Y
                    self._draw_card(self.screen, card, tx, cy)

        # 5. Render Active Card in Flight
        if self.current_animation:
            cur_x, cur_y = self.current_animation.get_current_pos()
            for idx, c in enumerate(self.current_animation.cards):
                card_y = cur_y + idx * self.TABLEAU_OFFSET_Y
                # Drop shadow
                shadow_rect = pygame.Rect(cur_x + 6, card_y + 8, self.CARD_W, self.CARD_H)
                pygame.draw.rect(self.screen, (10, 20, 15), shadow_rect, border_radius=self.CARD_CORNER)
                # Card
                self._draw_card(self.screen, c, cur_x, card_y)
                # Golden aura outline
                pygame.draw.rect(self.screen, (245, 158, 11), pygame.Rect(cur_x, card_y, self.CARD_W, self.CARD_H), 2, border_radius=self.CARD_CORNER)

        # 7. Bottom Action Log Bar
        bot_bar_w = self.BOARD_WIDTH if self.show_overlay else self.WIDTH
        bot_bar_rect = pygame.Rect(0, self.HEIGHT - 36, bot_bar_w, 36)
        pygame.draw.rect(self.screen, (12, 38, 24), bot_bar_rect)
        action_surf = self.font_small.render(f"Last Action: {self.last_action_text}", True, (220, 240, 230))
        self.screen.blit(action_surf, (20, self.HEIGHT - 26))

        # 8. Render Laya Decision Log Overlay on the right
        self._draw_overlay()

        pygame.display.flip()

    @property
    def tableaus(self):
        return self.engine.tableaus


# ==========================================
# Headless Batch Collector
# ==========================================

def run_headless_collection(num_episodes: int = 5, max_steps: int = 120):
    """Runs autonomous games headlessly with loop and stall filtering to generate high-quality datasets."""
    print(f"\n[Data Collector] Starting headless collection for {num_episodes} verified quality episodes...")
    engine = SolitaireEngine()
    logger = DataLogger(filepath="dataset/laya_dataset.jsonl", filter_beneficial=False)
    agent = LayaAgent(use_heuristic=True)

    successful_episodes = 0
    attempt = 0
    max_attempts = max(num_episodes * 6, 20)

    while successful_episodes < num_episodes and attempt < max_attempts:
        attempt += 1
        engine.reset()
        logger.start_episode()

        for step in range(max_steps):
            if engine.is_won():
                break
            if engine.is_loop_detected() or engine.is_stalled(25):
                break

            context = engine.get_context_summary()
            valid_actions = engine.get_valid_actions()
            options = [act.to_option_string() for act in valid_actions]

            if not options or (len(options) == 1 and options[0] == "No Action"):
                break

            best_option, is_beneficial = agent.select_action(context, options)
            executed = engine.execute_option_string(best_option)

            if executed:
                logger.log_step(context, options, best_option, is_beneficial)
            else:
                break

        won = engine.is_won()
        found_cards = engine.get_total_foundation_cards()

        # Commit if game won or made meaningful foundation progress (>= 2 cards)
        if won or found_cards >= 2:
            committed, count = logger.commit_episode(won=won, foundation_progress=found_cards, min_progress_required=2)
            if committed:
                successful_episodes += 1
                print(f" [Quality Episode {successful_episodes}/{num_episodes}] COMMITTED {count} steps (Foundations: {found_cards} cards)")
        else:
            _, dropped = logger.discard_episode(reason="loop_or_stall")
            print(f" [Attempt {attempt}] Stalled/Loop detected -> Discarded {dropped} steps (Dataset protected)")

    print(f"\n[Data Collector] Finished! Stats: {logger.stats}")
    print(f"[Data Collector] Total clean dataset records: {logger.count_records()} in {logger.filepath}\n")


# ==========================================
# Main CLI Entrypoint
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Laya Solitaire Environment")
    parser.add_argument("--headless", action="store_true", help="Run without Pygame UI")
    parser.add_argument("--collect", type=int, default=0, help="Number of episodes to auto-collect data headlessly")
    parser.add_argument("--bot", action="store_true", default=True, help="Start in Bot play mode")
    parser.add_argument("--human", action="store_true", help="Start in Human play mode")
    parser.add_argument("--delay", type=float, default=0.4, help="Bot turn delay in seconds (default: 0.4s)")
    parser.add_argument("--no_log", action="store_true", help="Disable turn logging")
    parser.add_argument("--model", type=str, default="heuristic", help="Agent model ('heuristic' or HF model path)")

    args = parser.parse_args()

    if args.headless or args.collect > 0:
        episodes = args.collect if args.collect > 0 else 5
        run_headless_collection(num_episodes=episodes)
        return

    gui = SolitaireGUI(
        bot_mode=not args.human,
        bot_delay=args.delay,
        enable_logging=not args.no_log,
        use_heuristic=(args.model == "heuristic"),
        model_path=args.model
    )
    gui.run()


if __name__ == "__main__":
    main()
