import numpy as np
import matplotlib.pyplot as plt

class Domain:

    def __init__(self, grid_length, grid_width):
        """Creates the grid world of desired dimensions and initalizes the state variable and time step count."""
        self.grid_length = grid_length
        self.grid_width = grid_width
        self.current_state = np.array([0,0])
        self.num_steps = 0
        self.seed = np.random.seed(np.random.randint(0,1000000))
    
    def reset_env(self):
        """Resets the current state to a random, non-terminal, state and resets the time step counter."""
        self.current_state = np.array([np.random.randint(0, self.grid_length), np.random.randint(0, self.grid_width)])
        self.num_steps = 0
        while self.is_terminal():
            self.current_state = np.array([np.random.randint(0, self.grid_length), np.random.randint(0, self.grid_width)])
        return np.copy(self.current_state), False
    
    def get_reward(self):
        """Returns the reward for being in a given state."""
        if np.array_equal(self.current_state, [self.grid_length-1,self.grid_width-1]):
            return 1.0
        else:
            return -0.01

    # Actions: 0=up, 1=left, 2=down, 3=right
    def get_next_state(self, state, action):
        """Returns the next state in the grid world given any state and action pair. Does not update the actual 
        game world and so can be used to just query any state action pair."""
        new_coords = np.copy(state)
        if action == 0: # Try go up
            if state[0] != 0:
                new_coords[0] = new_coords[0] - 1
        elif action == 2: # Try go down
            if state[0] != self.grid_length-1:
                new_coords[0] = new_coords[0] + 1
        elif action == 1: # Try go left
            if state[1] != 0:
                new_coords[1] = new_coords[1] - 1
        elif action == 3: # Try go left
            if state[1] != self.grid_width - 1:
                new_coords[1] = new_coords[1] + 1
        return new_coords
    
    def is_terminal(self):
        """Returns true if a terminal state is reached by the agent or the time step counter exceeds 15 steps."""
        if np.array_equal(self.current_state, [self.grid_length-1,self.grid_width-1]) or (self.num_steps>15):
            return True
        else:
            return False

    def take_action(self, action):
        """Used to get an agent to act within the environment. Updates the all variables as a result of the action
        and returns the reward received, next state and whether or not a terminal state was reached."""
        reward_out = self.get_reward()
        is_term = self.is_terminal()
        self.current_state = self.get_next_state(self.current_state, action)
        self.num_steps = self.num_steps + 1
        return reward_out, np.copy(self.current_state), is_term

    def print_state(self):
        """Displays the current game state as an image."""
        state_image = np.zeros((self.grid_length, self.grid_width))
        state_image[self.current_state[0], self.current_state[1]] = 1
        state_image[self.grid_length-1, self.grid_width-1] = 2
        plt.imshow(state_image, cmap='inferno')
        plt.axis('off')
        plt.show()
