"""
Risk Game Engine v3
Complete 42-territory implementation with neural network support
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum
import random

class Phase(Enum):
    DEPLOY = "deploy"
    ATTACK = "attack"
    FORTIFY = "fortify"

@dataclass
class Territory:
    name: str
    continent: str
    neighbors: List[str]
    owner: Optional[int] = None
    armies: int = 0

@dataclass
class GameState:
    territories: Dict[str, Territory]
    players: List[int]
    current_player: int
    phase: Phase
    cards: Dict[int, List[str]]
    turn_number: int = 0
    conquered_this_turn: bool = False  # Track if player conquered a territory

class RiskGameEnvironment:
    """Complete Risk game environment with 42 territories"""
    
    def __init__(self):
        self.territories = self._create_territories()
        self.reset()
    
    def _create_territories(self) -> Dict[str, Territory]:
        """Create the complete 42-territory Risk map"""
        territories = {
            # North America (9 territories)
            "alaska": Territory("alaska", "north_america", ["northwest_territory", "alberta", "kamchatka"]),
            "northwest_territory": Territory("northwest_territory", "north_america", ["alaska", "alberta", "ontario", "greenland"]),
            "alberta": Territory("alberta", "north_america", ["alaska", "northwest_territory", "ontario", "western_usa"]),
            "ontario": Territory("ontario", "north_america", ["northwest_territory", "alberta", "western_usa", "eastern_usa", "quebec", "greenland"]),
            "quebec": Territory("quebec", "north_america", ["ontario", "eastern_usa", "greenland"]),
            "western_usa": Territory("western_usa", "north_america", ["alberta", "ontario", "eastern_usa", "central_america"]),
            "eastern_usa": Territory("eastern_usa", "north_america", ["ontario", "quebec", "western_usa", "central_america"]),
            "central_america": Territory("central_america", "north_america", ["western_usa", "eastern_usa", "venezuela"]),
            "greenland": Territory("greenland", "north_america", ["northwest_territory", "ontario", "quebec", "iceland"]),
            
            # South America (4 territories)
            "venezuela": Territory("venezuela", "south_america", ["central_america", "brazil", "peru"]),
            "brazil": Territory("brazil", "south_america", ["venezuela", "peru", "argentina", "north_africa"]),
            "peru": Territory("peru", "south_america", ["venezuela", "brazil", "argentina"]),
            "argentina": Territory("argentina", "south_america", ["peru", "brazil"]),
            
            # Europe (7 territories)
            "iceland": Territory("iceland", "europe", ["greenland", "great_britain", "scandinavia"]),
            "great_britain": Territory("great_britain", "europe", ["iceland", "scandinavia", "northern_europe", "western_europe"]),
            "scandinavia": Territory("scandinavia", "europe", ["iceland", "great_britain", "northern_europe", "ukraine"]),
            "northern_europe": Territory("northern_europe", "europe", ["great_britain", "scandinavia", "ukraine", "southern_europe", "western_europe"]),
            "western_europe": Territory("western_europe", "europe", ["great_britain", "northern_europe", "southern_europe", "north_africa"]),
            "southern_europe": Territory("southern_europe", "europe", ["western_europe", "northern_europe", "ukraine", "middle_east", "egypt", "north_africa"]),
            "ukraine": Territory("ukraine", "europe", ["scandinavia", "northern_europe", "southern_europe", "ural", "afghanistan", "middle_east"]),
            
            # Africa (6 territories)
            "north_africa": Territory("north_africa", "africa", ["western_europe", "southern_europe", "egypt", "east_africa", "congo", "brazil"]),
            "egypt": Territory("egypt", "africa", ["southern_europe", "middle_east", "east_africa", "north_africa"]),
            "east_africa": Territory("east_africa", "africa", ["egypt", "middle_east", "north_africa", "congo", "south_africa", "madagascar"]),
            "congo": Territory("congo", "africa", ["north_africa", "east_africa", "south_africa"]),
            "south_africa": Territory("south_africa", "africa", ["congo", "east_africa", "madagascar"]),
            "madagascar": Territory("madagascar", "africa", ["east_africa", "south_africa"]),
            
            # Asia (12 territories)
            "middle_east": Territory("middle_east", "asia", ["southern_europe", "ukraine", "afghanistan", "india", "east_africa", "egypt"]),
            "afghanistan": Territory("afghanistan", "asia", ["ukraine", "ural", "china", "india", "middle_east"]),
            "ural": Territory("ural", "asia", ["ukraine", "siberia", "china", "afghanistan"]),
            "siberia": Territory("siberia", "asia", ["ural", "yakutsk", "irkutsk", "mongolia", "china"]),
            "yakutsk": Territory("yakutsk", "asia", ["siberia", "kamchatka", "irkutsk"]),
            "kamchatka": Territory("kamchatka", "asia", ["yakutsk", "irkutsk", "mongolia", "japan", "alaska"]),
            "irkutsk": Territory("irkutsk", "asia", ["siberia", "yakutsk", "kamchatka", "mongolia"]),
            "mongolia": Territory("mongolia", "asia", ["siberia", "irkutsk", "kamchatka", "japan", "china"]),
            "japan": Territory("japan", "asia", ["kamchatka", "mongolia"]),
            "china": Territory("china", "asia", ["ural", "siberia", "mongolia", "afghanistan", "india", "siam"]),
            "india": Territory("india", "asia", ["middle_east", "afghanistan", "china", "siam"]),
            "siam": Territory("siam", "asia", ["india", "china", "indonesia"]),
            
            # Australia (4 territories)
            "indonesia": Territory("indonesia", "australia", ["siam", "new_guinea", "western_australia"]),
            "new_guinea": Territory("new_guinea", "australia", ["indonesia", "western_australia", "eastern_australia"]),
            "western_australia": Territory("western_australia", "australia", ["indonesia", "new_guinea", "eastern_australia"]),
            "eastern_australia": Territory("eastern_australia", "australia", ["new_guinea", "western_australia"])
        }
        return territories
    
    def _initial_setup(self):
        """Properly assign territories and place initial armies like real Risk"""
        territory_names = list(self.territories.keys())
        random.shuffle(territory_names)
        
        # Assign territories to players in round-robin fashion
        for i, territory_name in enumerate(territory_names):
            player = i % len(self.game_state.players)
            self.territories[territory_name].owner = player
            self.territories[territory_name].armies = 1  # Start with 1 army per territory
        
        # Give each player additional armies to deploy (like real Risk setup)
        additional_armies_per_player = 5  # Each player gets 5 extra armies to place
        
        for player in self.game_state.players:
            # Get territories owned by this player
            owned_territories = [name for name, territory in self.territories.items() 
                               if territory.owner == player]
            
            # Randomly distribute additional armies
            for _ in range(additional_armies_per_player):
                random_territory = random.choice(owned_territories)
                self.territories[random_territory].armies += 1
    
    def reset(self):
        """Reset game to initial state"""
        self.game_state = GameState(
            territories=self.territories,
            players=[0, 1, 2, 3],  # 4 players
            current_player=0,
            phase=Phase.DEPLOY,
            cards={i: [] for i in range(4)},
            turn_number=0,
            conquered_this_turn=False
        )
        self._initial_setup()
        
        # Initialize card deck (Infantry, Cavalry, Artillery)
        self.card_deck = []
        card_types = ['Infantry', 'Cavalry', 'Artillery']
        for territory_name in self.territories.keys():
            card_type = card_types[len(self.card_deck) % 3]
            self.card_deck.append((territory_name, card_type))
        random.shuffle(self.card_deck)
    
    def get_state_vector(self) -> np.ndarray:
        """Convert game state to neural network input vector"""
        state_size = len(self.territories) * 6 + 20
        state = np.zeros(state_size)
        
        idx = 0
        # Territory information
        for territory in self.territories.values():
            state[idx] = territory.owner if territory.owner is not None else -1
            state[idx + 1] = territory.armies
            state[idx + 2] = hash(territory.continent) % 10
            state[idx + 3] = len(territory.neighbors)
            state[idx + 4] = sum(hash(n) % 100 for n in territory.neighbors) / 100
            state[idx + 5] = 1 if territory.owner == self.game_state.current_player else 0
            idx += 6
        
        # Game phase and player info
        state[idx] = self.game_state.phase.value == "deploy"
        state[idx + 1] = self.game_state.phase.value == "attack"
        state[idx + 2] = self.game_state.phase.value == "fortify"
        state[idx + 3] = self.game_state.current_player
        state[idx + 4] = self.game_state.turn_number / 100
        
        # Player army counts
        for player in range(4):
            player_armies = sum(t.armies for t in self.territories.values() if t.owner == player)
            state[idx + 5 + player] = player_armies / 50
        
        # Card counts
        for player in range(4):
            state[idx + 9 + player] = len(self.game_state.cards[player]) / 10
        
        return state

class RiskNeuralNetwork(nn.Module):
    """Neural network for playing Risk"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(RiskNeuralNetwork, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        shared_features = self.shared_layers(state)
        value = self.value_head(shared_features)
        policy = self.policy_head(shared_features)
        return value, policy

class RiskAgent:
    """AI agent that plays Risk using neural network"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = RiskNeuralNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = []
        self.memory_size = 10000
        
        # Training metrics
        self.losses = []
        self.episode_rewards = []
    
    def get_action(self, state: np.ndarray, valid_actions: List[int], epsilon: float = 0.1):
        """Get action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value, policy = self.network(state_tensor)
        
        # Mask invalid actions
        policy_np = policy.cpu().numpy()[0]
        masked_policy = np.zeros_like(policy_np)
        for action in valid_actions:
            if action < len(masked_policy):
                masked_policy[action] = policy_np[action]
        
        # Renormalize
        if masked_policy.sum() > 0:
            masked_policy /= masked_policy.sum()
            return np.random.choice(len(masked_policy), p=masked_policy)
        else:
            return random.choice(valid_actions)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        # Ensure states are numpy arrays
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size: int = 32):
        """Train the neural network"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # Convert to numpy arrays first
        states_np = np.array([e[0] for e in batch], dtype=np.float32)
        actions_np = np.array([e[1] for e in batch], dtype=np.int64)
        rewards_np = np.array([e[2] for e in batch], dtype=np.float32)
        next_states_np = np.array([e[3] for e in batch], dtype=np.float32)
        dones_np = np.array([e[4] for e in batch], dtype=bool)
        
        # Convert to tensors
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        
        # Current Q values
        current_values, current_policies = self.network(states)
        
        # Next Q values
        with torch.no_grad():
            next_values, _ = self.network(next_states)
            target_values = rewards + (0.99 * next_values.squeeze() * ~dones)
        
        # Value loss
        value_loss = self.criterion(current_values.squeeze(), target_values)
        
        # Policy loss
        action_probs = current_policies.gather(1, actions.unsqueeze(1))
        policy_loss = -torch.log(action_probs.squeeze() + 1e-8) * (target_values - current_values.squeeze().detach())
        policy_loss = policy_loss.mean()
        
        total_loss = value_loss + policy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.losses.append(total_loss.item())
        
        return total_loss.item()

class RiskActionSpace:
    """Defines the action space for Risk"""
    
    def __init__(self, num_territories: int):
        self.num_territories = num_territories
        self.action_size = num_territories * 5 + 1
    
    def get_valid_actions(self, game_state: GameState, player_id: int) -> List[int]:
        """Get list of valid actions"""
        valid_actions = []
        
        if game_state.phase == Phase.DEPLOY:
            for i, territory in enumerate(game_state.territories.values()):
                if territory.owner == player_id:
                    valid_actions.append(i)
        
        elif game_state.phase == Phase.ATTACK:
            for i, territory in enumerate(game_state.territories.values()):
                if territory.owner == player_id and territory.armies > 1:
                    for neighbor_name in territory.neighbors:
                        neighbor = game_state.territories[neighbor_name]
                        if neighbor.owner != player_id:
                            valid_actions.append(self.num_territories + i)
            valid_actions.append(self.action_size - 1)
        
        elif game_state.phase == Phase.FORTIFY:
            for i, territory in enumerate(game_state.territories.values()):
                if territory.owner == player_id and territory.armies > 1:
                    valid_actions.append(self.num_territories * 2 + i)
            valid_actions.append(self.action_size - 1)
        
        return valid_actions if valid_actions else [self.action_size - 1]

def game_over(game_state: GameState) -> bool:
    """Check if the game is over"""
    players_with_territories = set()
    total_territories = len(game_state.territories)
    
    for territory in game_state.territories.values():
        if territory.owner is not None:
            players_with_territories.add(territory.owner)
    
    if len(players_with_territories) <= 1:
        return True
    
    for player in players_with_territories:
        territories_owned = sum(1 for t in game_state.territories.values() if t.owner == player)
        if territories_owned >= (total_territories * 0.75):
            return True
    
    if game_state.turn_number > 100:
        return True
    
    return False

def get_game_winner(game_state: GameState) -> int:
    """Determine the winner"""
    territory_counts = {}
    army_counts = {}
    
    for territory in game_state.territories.values():
        if territory.owner is not None:
            territory_counts[territory.owner] = territory_counts.get(territory.owner, 0) + 1
            army_counts[territory.owner] = army_counts.get(territory.owner, 0) + territory.armies
    
    if not territory_counts:
        return 0
    
    max_territories = max(territory_counts.values())
    territory_leaders = [player for player, count in territory_counts.items() if count == max_territories]
    
    if len(territory_leaders) == 1:
        return territory_leaders[0]
    else:
        return max(territory_leaders, key=lambda p: army_counts.get(p, 0))

def calculate_end_game_rewards(game_state: GameState) -> List[float]:
    """Calculate rewards based on final game state"""
    territory_counts = [0] * 4
    army_counts = [0] * 4
    
    for territory in game_state.territories.values():
        if territory.owner is not None:
            territory_counts[territory.owner] += 1
            army_counts[territory.owner] += territory.armies
    
    total_territories = sum(territory_counts)
    winner = get_game_winner(game_state)
    
    rewards = [0.0] * 4
    
    for player in range(4):
        territory_ratio = territory_counts[player] / max(1, total_territories)
        rewards[player] = territory_ratio * 10
        
        army_ratio = army_counts[player] / max(1, sum(army_counts))
        rewards[player] += army_ratio * 5
        
        if player == winner:
            if territory_counts[player] >= total_territories * 0.9:
                rewards[player] += 50
            elif territory_counts[player] >= total_territories * 0.7:
                rewards[player] += 30
            else:
                rewards[player] += 15
        
        if territory_counts[player] == 0:
            rewards[player] = -20
    
    return rewards

def execute_action(env: RiskGameEnvironment, action: int) -> Tuple[np.ndarray, float, bool]:
    """Execute an action in the environment"""
    game_state = env.game_state
    current_player = game_state.current_player
    reward = 0
    
    action_space = RiskActionSpace(len(env.territories))
    valid_actions = action_space.get_valid_actions(game_state, current_player)
    
    if action not in valid_actions:
        reward = -0.5
        _advance_game_state(env)
        return env.get_state_vector(), reward, game_over(env.game_state)
    
    if game_state.phase == Phase.DEPLOY:
        reward = _execute_deploy_action(env, action, current_player)
    elif game_state.phase == Phase.ATTACK:
        reward = _execute_attack_action(env, action, current_player)
    elif game_state.phase == Phase.FORTIFY:
        reward = _execute_fortify_action(env, action, current_player)
    
    _advance_game_state(env)
    
    next_state = env.get_state_vector()
    done = game_over(env.game_state)
    
    return next_state, reward, done

def _execute_deploy_action(env: RiskGameEnvironment, action: int, player: int) -> float:
    """Execute deployment action"""
    owned_territories = [name for name, territory in env.territories.items() if territory.owner == player]
    if owned_territories:
        target_territory = random.choice(owned_territories)
        env.territories[target_territory].armies += 1
        return 0.1
    return -0.1

def _execute_attack_action(env: RiskGameEnvironment, action: int, player: int) -> float:
    """Execute attack action"""
    reward = 0
    attack_candidates = []
    
    for name, territory in env.territories.items():
        if territory.owner == player and territory.armies > 1:
            for neighbor_name in territory.neighbors:
                neighbor = env.territories[neighbor_name]
                if neighbor.owner != player:
                    attack_candidates.append((name, neighbor_name))
    
    if attack_candidates:
        attacker_territory, defender_territory = random.choice(attack_candidates)
        
        attacker_armies = env.territories[attacker_territory].armies
        defender_armies = env.territories[defender_territory].armies
        defender_player = env.territories[defender_territory].owner
        
        attack_strength = attacker_armies + random.uniform(0, 2)
        defense_strength = defender_armies + random.uniform(0, 1.5)
        
        if attack_strength > defense_strength:
            # Conquest successful!
            env.territories[defender_territory].owner = player
            env.territories[defender_territory].armies = max(1, attacker_armies // 2)
            env.territories[attacker_territory].armies = max(1, attacker_armies - attacker_armies // 2)
            
            # Mark that player conquered a territory this turn
            env.game_state.conquered_this_turn = True
            
            reward = 2.0
            
            remaining_territories = sum(1 for t in env.territories.values() if t.owner == defender_player)
            if remaining_territories == 0:
                reward += 10.0
            
            player_territories = sum(1 for t in env.territories.values() if t.owner == player)
            total_territories = len(env.territories)
            dominance_bonus = (player_territories / total_territories) * 5
            reward += dominance_bonus
        else:
            losses = min(attacker_armies - 1, random.randint(1, 2))
            env.territories[attacker_territory].armies -= losses
            
            if random.random() < 0.3:
                env.territories[defender_territory].armies = max(1, defender_armies - 1)
            
            reward = -0.5
    
    return reward

def _execute_fortify_action(env: RiskGameEnvironment, action: int, player: int) -> float:
    """Execute fortify action"""
    owned_territories = [name for name, territory in env.territories.items() 
                        if territory.owner == player and territory.armies > 1]
    
    if owned_territories:
        source_territory = random.choice(owned_territories)
        source = env.territories[source_territory]
        
        owned_neighbors = [neighbor for neighbor in source.neighbors 
                          if env.territories[neighbor].owner == player]
        
        if owned_neighbors:
            target_territory = random.choice(owned_neighbors)
            env.territories[source_territory].armies -= 1
            env.territories[target_territory].armies += 1
            return 0.05
    
    return 0

def _advance_game_state(env: RiskGameEnvironment):
    """Advance the game state to next phase/player"""
    game_state = env.game_state
    
    if game_state.phase == Phase.DEPLOY:
        # Check if player can/wants to trade in cards for reinforcements
        _attempt_card_trade(env, game_state.current_player)
        game_state.phase = Phase.ATTACK
    elif game_state.phase == Phase.ATTACK:
        game_state.phase = Phase.FORTIFY
    elif game_state.phase == Phase.FORTIFY:
        # Give card if player conquered at least one territory this turn
        if game_state.conquered_this_turn and len(env.card_deck) > 0:
            card = env.card_deck.pop()
            game_state.cards[game_state.current_player].append(card)
            print(f"Player {game_state.current_player + 1} earned a card: {card[1]} ({card[0]})")
        
        # Reset conquest flag
        game_state.conquered_this_turn = False
        
        # Move to next player
        game_state.phase = Phase.DEPLOY
        game_state.current_player = (game_state.current_player + 1) % len(game_state.players)
        game_state.turn_number += 1
        _give_reinforcements(env, game_state.current_player)

def _attempt_card_trade(env: RiskGameEnvironment, player: int):
    """Attempt to trade in cards for reinforcements"""
    cards = env.game_state.cards[player]
    
    # Need at least 3 cards to trade
    if len(cards) < 3:
        return
    
    # Check for valid sets
    card_types = [card[1] for card in cards]
    
    # Check for three of a kind
    for card_type in ['Infantry', 'Cavalry', 'Artillery']:
        if card_types.count(card_type) >= 3:
            _trade_in_cards(env, player, card_type)
            return
    
    # Check for one of each
    if 'Infantry' in card_types and 'Cavalry' in card_types and 'Artillery' in card_types:
        _trade_in_cards(env, player, 'set')
        return
    
    # Force trade if player has 5+ cards (Risk rule)
    if len(cards) >= 5:
        # Trade first 3 cards regardless
        _trade_in_cards(env, player, 'forced')

def _trade_in_cards(env: RiskGameEnvironment, player: int, trade_type: str):
    """Trade in 3 cards for reinforcements"""
    cards = env.game_state.cards[player]
    
    # Remove 3 cards based on trade type
    if trade_type in ['Infantry', 'Cavalry', 'Artillery']:
        # Three of a kind
        removed = 0
        for i in range(len(cards) - 1, -1, -1):
            if cards[i][1] == trade_type:
                cards.pop(i)
                removed += 1
                if removed == 3:
                    break
    elif trade_type == 'set':
        # One of each
        for card_type in ['Infantry', 'Cavalry', 'Artillery']:
            for i in range(len(cards) - 1, -1, -1):
                if cards[i][1] == card_type:
                    cards.pop(i)
                    break
    else:  # forced
        # Remove first 3 cards
        for _ in range(3):
            if cards:
                cards.pop(0)
    
    # Calculate reinforcements (standard Risk: 4, 6, 8, 10, 12, 15, then +5 each time)
    # Simplified: start at 4 and increase by 2 each trade
    trades_made = sum(len(env.game_state.cards[p]) < len(cards) for p in range(4))
    reinforcements = 4 + (trades_made * 2)
    
    # Place reinforcements on owned territories
    owned_territories = [name for name, territory in env.territories.items() if territory.owner == player]
    
    if owned_territories:
        print(f"Player {player + 1} traded cards for {reinforcements} reinforcements!")
        for _ in range(reinforcements):
            target = random.choice(owned_territories)
            env.territories[target].armies += 1

def _give_reinforcements(env: RiskGameEnvironment, player: int):
    """Give reinforcement armies to player"""
    territories_owned = sum(1 for t in env.territories.values() if t.owner == player)
    reinforcements = max(3, territories_owned // 3)
    
    owned_territories = [name for name, territory in env.territories.items() if territory.owner == player]
    
    for _ in range(reinforcements):
        if owned_territories:
            target = random.choice(owned_territories)
            env.territories[target].armies += 1
