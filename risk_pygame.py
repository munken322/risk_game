import pygame
import sys
import math
from typing import Dict, Tuple, List
import numpy as np

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
FPS = 60

# Colors
BACKGROUND_COLOR = (240, 248, 255)  # Alice Blue
OCEAN_COLOR = (135, 206, 250)  # Light Sky Blue
TEXT_COLOR = (0, 0, 0)
BORDER_COLOR = (100, 100, 100)

# Player colors (vibrant)
PLAYER_COLORS = [
    (255, 107, 107),  # Red
    (78, 205, 196),   # Teal
    (69, 183, 209),   # Blue
    (150, 206, 180),  # Green
]

# Continent colors (pastel backgrounds)
CONTINENT_COLORS = {
    'north_america': (255, 204, 204, 100),
    'south_america': (255, 255, 204, 100),
    'europe': (204, 255, 204, 100),
    'africa': (255, 204, 153, 100),
    'asia': (255, 204, 255, 100),
    'australia': (204, 204, 255, 100)
}

class PygameRiskVisualizer:
    def __init__(self, game_env):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Risk Neural Network Training")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        self.game_env = game_env
        self.running = True
        self.paused = False
        
        # Layout for game board (left side)
        self.board_rect = pygame.Rect(20, 100, 900, 750)
        
        # Training stats area (right side)
        self.stats_rect = pygame.Rect(940, 100, 440, 750)
        
        # Create territory positions
        self.territory_positions = self._create_territory_layout()
        
        # Training statistics
        self.episode = 0
        self.episode_history = []
        self.win_counts = [0, 0, 0, 0]
        self.territory_history = []
        self.current_turn = 0
        
    def _create_territory_layout(self) -> Dict[str, Tuple[int, int]]:
        """Create pixel coordinates for each territory on the game board"""
        # Scale positions to fit in board area
        base_positions = {
            # North America
            'alaska': (-0.8, 0.7),
            'alberta': (-0.7, 0.5),
            'ontario': (-0.5, 0.5),
            'quebec': (-0.4, 0.4),
            'western_usa': (-0.7, 0.3),
            'eastern_usa': (-0.5, 0.3),
            'central_america': (-0.6, 0.1),
            'greenland': (-0.2, 0.8),
            
            # South America
            'venezuela': (-0.5, -0.1),
            'brazil': (-0.3, -0.3),
            'peru': (-0.5, -0.4),
            'argentina': (-0.4, -0.6),
            
            # Europe
            'iceland': (-0.1, 0.6),
            'great_britain': (-0.05, 0.5),
            'scandinavia': (0.1, 0.7),
            'northern_europe': (0.1, 0.5),
            'western_europe': (0.0, 0.4),
            'southern_europe': (0.1, 0.3),
            'ukraine': (0.3, 0.5),
            
            # Africa
            'north_africa': (0.1, 0.1),
            'egypt': (0.2, 0.2),
            'east_africa': (0.3, 0.0),
            'congo': (0.2, -0.2),
            'south_africa': (0.2, -0.4),
            'madagascar': (0.4, -0.3),
            
            # Asia
            'middle_east': (0.3, 0.3),
            'afghanistan': (0.5, 0.3),
            'ural': (0.4, 0.6),
            'siberia': (0.6, 0.7),
            'yakutsk': (0.8, 0.8),
            'kamchatka': (0.9, 0.6),
            'irkutsk': (0.7, 0.5),
            'mongolia': (0.7, 0.3),
            'japan': (0.9, 0.3),
            'china': (0.6, 0.2),
            'india': (0.5, 0.1),
            'siam': (0.6, 0.0),
            
            # Australia
            'indonesia': (0.7, -0.2),
            'new_guinea': (0.8, -0.3),
            'western_australia': (0.8, -0.5),
            'eastern_australia': (0.9, -0.4)
        }
        
        # Convert to pixel coordinates
        pixel_positions = {}
        center_x = self.board_rect.centerx
        center_y = self.board_rect.centery
        scale = 350
        
        for territory, (x, y) in base_positions.items():
            pixel_x = center_x + int(x * scale)
            pixel_y = center_y - int(y * scale)  # Flip Y axis
            pixel_positions[territory] = (pixel_x, pixel_y)
        
        return pixel_positions
    
    def draw_board(self, game_state):
        """Draw the Risk game board"""
        # Draw ocean background
        pygame.draw.rect(self.screen, OCEAN_COLOR, self.board_rect)
        
        # Draw continent backgrounds
        self._draw_continent_backgrounds(game_state)
        
        # Draw territory connections
        self._draw_territory_connections(game_state)
        
        # Draw territories
        self._draw_territories(game_state)
        
        # Draw board border
        pygame.draw.rect(self.screen, BORDER_COLOR, self.board_rect, 3)
    
    def _draw_continent_backgrounds(self, game_state):
        """Draw colored backgrounds for each continent"""
        continents_territories = {}
        
        for territory_name, territory in game_state.territories.items():
            continent = territory.continent
            if continent not in continents_territories:
                continents_territories[continent] = []
            if territory_name in self.territory_positions:
                continents_territories[continent].append(self.territory_positions[territory_name])
        
        # Draw each continent
        for continent, positions in continents_territories.items():
            if continent in CONTINENT_COLORS and len(positions) >= 3:
                # Create convex hull or circle around territories
                center_x = sum(p[0] for p in positions) / len(positions)
                center_y = sum(p[1] for p in positions) / len(positions)
                
                # Calculate radius
                max_dist = max(math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in positions)
                radius = int(max_dist * 1.5)
                
                # Create surface with alpha
                surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                color = CONTINENT_COLORS[continent]
                pygame.draw.circle(surface, color, (radius, radius), radius)
                self.screen.blit(surface, (int(center_x - radius), int(center_y - radius)))
    
    def _draw_territory_connections(self, game_state):
        """Draw lines between connected territories"""
        for territory_name, territory in game_state.territories.items():
            if territory_name not in self.territory_positions:
                continue
            
            pos1 = self.territory_positions[territory_name]
            for neighbor in territory.neighbors:
                if neighbor in self.territory_positions:
                    pos2 = self.territory_positions[neighbor]
                    pygame.draw.line(self.screen, (180, 180, 180), pos1, pos2, 1)
    
    def _draw_territories(self, game_state):
        """Draw territory circles with armies"""
        for territory_name, territory in game_state.territories.items():
            if territory_name not in self.territory_positions:
                continue
            
            pos = self.territory_positions[territory_name]
            
            # Get color based on owner
            if territory.owner is not None:
                color = PLAYER_COLORS[territory.owner % len(PLAYER_COLORS)]
            else:
                color = (200, 200, 200)
            
            # Draw territory circle
            radius = 25
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 2)
            
            # Draw army count
            if territory.armies > 0:
                army_text = self.font_medium.render(str(territory.armies), True, (255, 255, 255))
                army_rect = army_text.get_rect(center=pos)
                
                # Black background circle for better visibility
                pygame.draw.circle(self.screen, (0, 0, 0), pos, 15)
                self.screen.blit(army_text, army_rect)
            
            # Draw territory name (smaller, below circle)
            name_display = territory_name.replace('_', ' ').title()
            if len(name_display) > 12:
                name_display = name_display[:10] + "."
            name_text = self.font_small.render(name_display, True, TEXT_COLOR)
            name_rect = name_text.get_rect(center=(pos[0], pos[1] + 35))
            self.screen.blit(name_text, name_rect)
    
    def draw_stats(self, game_state, agents, current_rewards):
        """Draw training statistics panel"""
        # Background
        pygame.draw.rect(self.screen, (255, 255, 255), self.stats_rect)
        pygame.draw.rect(self.screen, BORDER_COLOR, self.stats_rect, 3)
        
        y_offset = self.stats_rect.top + 20
        x_offset = self.stats_rect.left + 20
        
        # Title
        title = self.font_large.render("Training Stats", True, TEXT_COLOR)
        self.screen.blit(title, (x_offset, y_offset))
        y_offset += 50
        
        # Episode info
        episode_text = self.font_medium.render(f"Episode: {self.episode}", True, TEXT_COLOR)
        self.screen.blit(episode_text, (x_offset, y_offset))
        y_offset += 30
        
        turn_text = self.font_medium.render(f"Turn: {game_state.turn_number}", True, TEXT_COLOR)
        self.screen.blit(turn_text, (x_offset, y_offset))
        y_offset += 30
        
        phase_text = self.font_medium.render(f"Phase: {game_state.phase.value.title()}", True, TEXT_COLOR)
        self.screen.blit(phase_text, (x_offset, y_offset))
        y_offset += 40
        
        # Player statistics
        for i in range(4):
            # Player header with color
            pygame.draw.rect(self.screen, PLAYER_COLORS[i], 
                           (x_offset, y_offset, 30, 20))
            player_text = self.font_medium.render(f"Player {i+1}", True, TEXT_COLOR)
            self.screen.blit(player_text, (x_offset + 40, y_offset))
            y_offset += 25
            
            # Territory count
            territory_count = sum(1 for t in game_state.territories.values() if t.owner == i)
            army_count = sum(t.armies for t in game_state.territories.values() if t.owner == i)
            
            stats_text = self.font_small.render(
                f"  Territories: {territory_count}  Armies: {army_count}", 
                True, TEXT_COLOR
            )
            self.screen.blit(stats_text, (x_offset, y_offset))
            y_offset += 20
            
            # Win rate
            win_rate = self.win_counts[i] / max(1, self.episode)
            reward = current_rewards[i] if i < len(current_rewards) else 0
            
            performance_text = self.font_small.render(
                f"  Wins: {self.win_counts[i]}  Rate: {win_rate:.2%}  Reward: {reward:.1f}",
                True, TEXT_COLOR
            )
            self.screen.blit(performance_text, (x_offset, y_offset))
            y_offset += 35
        
        # Controls
        y_offset += 20
        controls_title = self.font_medium.render("Controls:", True, TEXT_COLOR)
        self.screen.blit(controls_title, (x_offset, y_offset))
        y_offset += 30
        
        controls = [
            "SPACE - Pause/Resume",
            "R - Reset Game",
            "Q - Quit",
            "UP/DOWN - Speed"
        ]
        
        for control in controls:
            control_text = self.font_small.render(control, True, TEXT_COLOR)
            self.screen.blit(control_text, (x_offset, y_offset))
            y_offset += 25
    
    def draw_header(self, current_player):
        """Draw header with game title and current player"""
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, 90)
        pygame.draw.rect(self.screen, (50, 50, 80), header_rect)
        
        # Title
        title = self.font_large.render("RISK: Neural Network Training", True, (255, 255, 255))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 30))
        self.screen.blit(title, title_rect)
        
        # Current player indicator
        if current_player is not None:
            player_color = PLAYER_COLORS[current_player % len(PLAYER_COLORS)]
            pygame.draw.rect(self.screen, player_color, (SCREEN_WIDTH//2 - 100, 50, 200, 30))
            current_text = self.font_medium.render(f"Current: Player {current_player + 1}", 
                                                   True, (255, 255, 255))
            current_rect = current_text.get_rect(center=(SCREEN_WIDTH // 2, 65))
            self.screen.blit(current_text, current_rect)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    return "reset"
        
        return True
    
    def update_display(self, game_state, agents=None, current_rewards=None):
        """Update the entire display"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw header
        self.draw_header(game_state.current_player)
        
        # Draw game board
        self.draw_board(game_state)
        
        # Draw statistics
        if agents and current_rewards:
            self.draw_stats(game_state, agents, current_rewards)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def update_stats(self, episode, winner, territory_counts):
        """Update training statistics"""
        self.episode = episode
        self.win_counts[winner] += 1
        self.territory_history.append(territory_counts.copy())
    
    def cleanup(self):
        """Cleanup pygame resources"""
        pygame.quit()

# Modified training function for pygame visualization
def train_with_pygame_visualization():
    """Train Risk agents with real-time Pygame visualization"""
    from risk_gamev2 import (RiskGameEnvironment, RiskAgent, RiskActionSpace, 
                             execute_action, game_over, get_game_winner, 
                             calculate_end_game_rewards)
    
    print("Starting Risk Neural Network Training with Pygame Visualization")
    print("=" * 70)
    
    # Initialize game environment
    env = RiskGameEnvironment()
    state_size = len(env.get_state_vector())
    action_space = RiskActionSpace(len(env.territories))
    
    # Create agents
    agents = [RiskAgent(state_size, action_space.action_size) for _ in range(4)]
    
    # Create pygame visualizer
    visualizer = PygameRiskVisualizer(env)
    
    # Training parameters
    num_episodes = 1000
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    # Visualization speed
    steps_per_frame = 5  # Process N steps before updating display
    
    episode = 0
    
    try:
        while visualizer.running and episode < num_episodes:
            # Handle events
            event_result = visualizer.handle_events()
            if not event_result:
                break
            elif event_result == "reset":
                env.reset()
                continue
            
            if visualizer.paused:
                visualizer.update_display(env.game_state, agents, [0]*4)
                continue
            
            # Start new episode
            env.reset()
            current_rewards = [0] * 4
            game_length = 0
            max_turns = 50
            
            # Play one game
            step_count = 0
            while visualizer.running and not game_over(env.game_state) and game_length < max_turns:
                current_player = env.game_state.current_player
                state = env.get_state_vector()
                valid_actions = action_space.get_valid_actions(env.game_state, current_player)
                
                # Get action
                action = agents[current_player].get_action(state, valid_actions, epsilon)
                
                # Execute action
                next_state, reward, done = execute_action(env, action)
                current_rewards[current_player] += reward
                
                # Store and train
                agents[current_player].store_experience(state, action, reward, next_state, done)
                if len(agents[current_player].memory) > 32:
                    agents[current_player].train()
                
                game_length += 1
                step_count += 1
                
                # Update display periodically
                if step_count >= steps_per_frame:
                    visualizer.update_display(env.game_state, agents, current_rewards)
                    step_count = 0
                
                # Handle events during gameplay
                if not visualizer.handle_events():
                    break
            
            # End of game
            end_game_rewards = calculate_end_game_rewards(env.game_state)
            for i in range(4):
                current_rewards[i] += end_game_rewards[i]
            
            winner = get_game_winner(env.game_state)
            territory_counts = [sum(1 for t in env.game_state.territories.values() if t.owner == i) 
                              for i in range(4)]
            
            # Update stats
            visualizer.update_stats(episode, winner, territory_counts)
            
            # Final display update for this episode
            visualizer.update_display(env.game_state, agents, current_rewards)
            
            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            episode += 1
            
            # Progress report
            if episode % 10 == 0:
                print(f"Episode {episode}: Winner P{winner+1}, Territories: {territory_counts}, " +
                      f"Epsilon: {epsilon:.3f}")
    
    finally:
        visualizer.cleanup()
    
    print("\nTraining complete!")
    return agents

if __name__ == "__main__":
    print("Risk Neural Network - Pygame Visualization")
    print("=" * 50)
    print("Make sure you have the main risk_gamev2.py file in the same directory!")
    print("\nControls:")
    print("  SPACE - Pause/Resume training")
    print("  R - Reset current game")
    print("  Q - Quit")
    print("\nStarting in 3 seconds...")
    
    import time
    time.sleep(3)
    
    train_with_pygame_visualization()
