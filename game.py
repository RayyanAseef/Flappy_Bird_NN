import math, random, pygame
# Goal is to design a flappy bird game with visuals being optional
# --> For bird will store its collision rect
# --> For pipe will store rect of passable area. When bird passes pipe new one replaces current.
# --> Constants: x velocity, jump initial velocity y, Gravity
# --> Variables: y velocity

class Flappy_Bird_Game:
    def __init__(self, game_width, game_height, easyMode=True):
        self.game_width = max(game_width, 1200)
        self.game_height = max(game_height, 600)

        self.score = 0

        # Setting Bird Params
        bird_length = min(50, self.game_width // 2, self.game_height // 2)
        self.bird_rect = pygame.Rect(max(150, int(self.game_width * 0.3)), random.randint(int(game_height * 0.4), int(game_height * 0.6)), bird_length, bird_length)

        # Setting Pipe Params
        self.pipes = []

        if easyMode:
            self.pipe_length = bird_length * 5
            self.pipe_gap = self.pipe_length * 2
            self.JUMP_VEL = 5

        else:
            self.pipe_length = bird_length * 3.5
            self.pipe_gap = self.pipe_length * 3
            self.JUMP_VEL = 8

        self.reset_pipes()

        self.X_VEL = 5
        self.GRAVITY = -5
        self.DELA_TIME = 0.1

        self.y_vel = 0

    def reset_pipes(self):
        self.pipes = []
        starting_pipe_count = math.ceil((self.game_width - (self.bird_rect.x + self.bird_rect.w + self.pipe_gap)) / float(self.pipe_gap + self.pipe_length))
        x = self.bird_rect.x + self.bird_rect.w + self.pipe_gap
        for i in range(starting_pipe_count):
            y = random.randint(int(self.game_height * 0.1), int(self.game_height * 0.9 - self.pipe_length))
            self.pipes.append(pygame.Rect(x, y, self.pipe_length, self.pipe_length))
            x += self.pipe_gap + self.pipe_length

    # Start New Game
    def reset(self):
        self.bird_rect = pygame.Rect(max(150, self.game_width * 0.3), random.randint(int(self.game_height * 0.4), int(self.game_height * 0.6)), self.bird_rect.w, self.bird_rect.h)
        self.reset_pipes()
        self.y_vel = 0
        self.score = 0

    
    def isDead(self):
        for pipe in self.pipes:
            if self.bird_rect.colliderect([pipe.x, 0, self.pipe_length, pipe.y]) or self.bird_rect.colliderect([pipe.x, pipe.y+self.pipe_length, self.pipe_length, self.game_height-(pipe.y+self.pipe_length)]):
                return True

        if self.bird_rect.y < 0 or self.bird_rect.y+self.bird_rect.h > self.game_height:
            return True
        
        return False

    def get_state(self):
        x_dist = float('inf')
        closest_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.w > self.bird_rect.x and (pipe.x + pipe.w - self.bird_rect.x) < x_dist:
                x_dist = pipe.x + pipe.w - self.bird_rect.x
                closest_pipe = pipe
        if closest_pipe is None:
            return self.bird_rect.y, self.y_vel, 0, 0, 0
        
        return self.y_vel, x_dist, self.bird_rect.y+self.bird_rect.h, closest_pipe.y, closest_pipe.y + closest_pipe.h

    def get_reward(self, action, prev_y, new_y):
        reward = -0.1  # Small negative reward for each step to encourage faster completion
        if self.isDead():
            reward = -1.0  # Large negative reward for dying
        else:
            for pipe in self.pipes:
                if pipe.x > self.bird_rect.y:
                    break
            if (action[1] == 1 and pipe.y + pipe.h > prev_y + self.bird_rect.h) or \
                    (action[0] == 1 and pipe.y < prev_y):
                reward = 0.5  # Positive reward for a good jump action
            if (prev_y < pipe.y and prev_y + self.bird_rect.h > pipe.y + pipe.h) and \
                    (new_y < pipe.y and new_y + self.bird_rect.h > pipe.y + pipe.h):
                reward = 1.0  # Large positive reward for passing through the pipe
        return reward

    # User/Ai input
    # Action is either jump( [0, 1] ) or nothing( [1, 0] )
    def input(self, action):
        if action[1] == 1:
            self.y_vel = self.JUMP_VEL

        for pipe in self.pipes:
            pipe.x -= self.X_VEL
            if pipe.x + pipe.w < 0:
                self.pipes.remove(pipe)
            if pipe.x + pipe.w < self.bird_rect.x < pipe.x + pipe.w + self.X_VEL+1:
                self.score += 1

        prev_y = self.bird_rect.y
        
        self.bird_rect.y -= self.y_vel
        self.y_vel += self.GRAVITY * self.DELA_TIME

        new_y = self.bird_rect.y


        if self.pipes[-1].x+self.pipe_length+self.pipe_gap < self.game_width:
            y = random.randint(self.game_height*0.1, self.game_height*0.9-self.pipe_length)
            self.pipes.append( pygame.Rect( self.pipes[-1].x+self.pipe_gap+self.pipe_length, y, self.pipe_length, self.pipe_length ) )

        return self.get_reward(action, prev_y, new_y)

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 0), self.bird_rect)

        for pipe in self.pipes:
            pygame.draw.rect(surface, (0, 255, 0), [pipe.x, 0, self.pipe_length, pipe.y])
            pygame.draw.rect(surface, (0, 255, 0), [pipe.x, pipe.y + self.pipe_length, self.pipe_length, self.game_height - (pipe.y + self.pipe_length)])

        # Draw the score
        font = pygame.font.Font(None, 74)
        text = font.render(str(self.score), True, (255, 255, 255))
        surface.blit(text, (10, 10))
