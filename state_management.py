class State:
    def __init__(self, all_states):
        #TODO Error handling for length < 1
        self.current_state = 0
        self.all_states = all_states
        self.num_states = len(self.all_states)
    
    def next(self):
        if self.current_state+1 < self.num_states:
            self.current_state = self.current_state+1
        
    def prev(self):
        if self.current_state-1 >= 0:
            self.current_state = self.current_state-1
    
    def reset(self):
        self.current_state = 0

    def get_current_state(self):
        return self.all_states[self.current_state]
    
