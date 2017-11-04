import numpy as np
import time
class JumperGame:
    def __init__(self):
        self.state = [[0]*5, [0,1], [1]*5]
        self.player_pos = [1,1]
        self.erase = '\x1b[1A\x1b[2K'
        self.fitness = 0
        self.last_action = -1
    def render(self):
        print 4*self.erase
        for index, i in enumerate(self.state[0]):
            p = ' ' if i == 0 else '@'
            if self.player_pos[0] == 0:
                if index == 1:
                    p = 'I'
            print p,
        print ""
        for index, i in enumerate(self.state[1]):
            p = ' '
            if self.player_pos[0] == 1:
                if index == 1:
                    p = 'I'
            print p,
        print ""
        for i in self.state[2]:
            p = ' ' if i == 0 else '@'
            print p,
        print ""
        print "fitness = {} ".format(self.fitness),

        # print ""
    def step(self, action=0):
        # print action
        t0 = np.random.choice(2,1,p=[0.8,0.2])[0]
        t1 = np.random.choice(2,1,p=[0.2,0.8])[0]
        state0 = self.state[0]
        state2 = self.state[2]
        # print action,
        if self.last_action == 1:
            action = 0
        self.player_pos = [1,1]
        if action == 0:
            if state2[2] == 0:
                # print "\n\nhole"
                return True
        if action == 1:
            self.player_pos = [0,1]
            if state0[2] == 1:
                # print "head hit"
                return True
        if t1 == 0: #for checking consecutive holes
            if state2[-1] == 0:
                t1 = 1
        if t0 == 1 and t1 == 0: #cannot jump or go
            t0 = 0

        state0 = state0[1:]
        state0.append(t0)
        state2 = state2[1:]
        state2.append(t1)
        new_state = [state0, self.state[1], state2]
        self.last_action = action
        # print t0
        # print t1
        self.state = new_state
        # print self.state,
        self.fitness = self.fitness + 1
        return False
    def get_input_to_algo(self):
        return self.state[0] + self.state[2]
    def get_fitness(self):
        return self.fitness
    def get_bottom_state(self):
        return self.state[2]
    def get_top_state(self):
        return self.state[0]
    def get_player_position(self):
        return self.player_pos;
