import pygame
import time
import tensorflow as tf
from optparse import OptionParser

from JumperGame import JumperGame

parser = OptionParser()
parser.add_option("-n", "--num-model", dest="MODEL_NUM",
                  help="The Model Number to User")

(options, args) = parser.parse_args()
print options, args
if options.MODEL_NUM == None:
    print "Give -n Model Number"
    exit()

pygame.init()
screen = pygame.display.set_mode((500, 600))
done = False
is_blue = True
x = 30
y = 30

clock = pygame.time.Clock()
pygame.display.set_caption('Jumper')

ball_img = pygame.image.load('assets/ball.png')
ball_img = pygame.transform.scale(ball_img, (50, 50))
spike_img = pygame.image.load('assets/spike.png')
spike_img = pygame.transform.scale(spike_img, (100, 100))
spike_img_inverted = pygame.transform.rotate(spike_img, 180)
MODEL_NUM = options.MODEL_NUM
MODEL_PREFIX = "./game_checkpoints/my_test_model-{}.meta"

sess = tf.Session()
new_saver = tf.train.import_meta_graph(MODEL_PREFIX.format(MODEL_NUM))
new_saver.restore(sess,MODEL_PREFIX.format(MODEL_NUM)[:-5])

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
game = JumperGame()


def ball(x=100,y=150):
    screen.blit(ball_img, (x,y))

def draw_top(x=0,state=[]):
    color = (255,128,0)
    X = 0
    for i in state:
        if i == 1:
            # pygame.draw.rect(screen, color, pygame.Rect(X-x, 0, 100-2, 100))
            screen.blit(spike_img_inverted, (X-x,0))
        X = X + 100

def draw_bottom(x=0, state=[]):
    color = (0,128,255)
    X = 0
    for i in state:
        if i == 1:
            pygame.draw.rect(screen, color, pygame.Rect(X-x, 200, 100-2, 100))
        else:
            screen.blit(spike_img, (X-x,200))
        X = X + 100

BALL_POS_MIDDLE = 150
BALL_POS_TOP = 50
INNER_FRAMES = 10

x1 = 0
curr_state_bottom = [1,1,1,1,1]
curr_state_top = [0,0,0,0,0]
ball_position = 1

last_ball_pos = BALL_POS_MIDDLE
font = pygame.font.SysFont("comicsansms", 38)
generation_text = font.render("Generation : {}".format(MODEL_NUM), True, (255, 255, 255))
def ball_movement_del(ball_position):
    to_position = BALL_POS_TOP if ball_position == 0 else BALL_POS_MIDDLE
    delta1 = float(last_ball_pos + (to_position - last_ball_pos) / INNER_FRAMES)
    return delta1
while not done:
    ip = game.get_input_to_algo()
    y_ = sess.run([tf.argmax(Y,1)], feed_dict={X:[ip]})[0]
    done = game.step(action=y_[0])
    state_bottom = game.get_bottom_state()
    state_top = game.get_top_state()
    ball_position = game.get_player_position()[0]
    fitness_text = font.render("Fitness : {}".format(game.get_fitness()), True, (255,255,255))
    over_text = font.render("Game Over", True, (255,0,0))
    x1 = 0

    for _ in range(INNER_FRAMES):
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        is_blue = not is_blue


        screen.fill((0, 0, 0))
        x1 = x1 + 100/INNER_FRAMES
        ball_del = ball_movement_del(ball_position)
        last_ball_pos = ball_del
        ball(y=ball_del)
        draw_bottom(x=x1, state=curr_state_bottom)
        draw_top(x=x1, state=curr_state_top)

        screen.blit(generation_text, (10, 340 - generation_text.get_height() // 2))
        screen.blit(fitness_text, (10, 380 - fitness_text.get_height() // 2))
        if done:
            screen.blit(over_text, (250 - over_text.get_width(), 150 - over_text.get_height() ))
        pygame.display.flip()
        clock.tick(90)
    curr_state_top = state_top
    curr_state_bottom = state_bottom
sess.close()
time.sleep(1)
