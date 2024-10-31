import pygame
import sys
from time import time
import random
from pygame.locals import *
from pygame.compat import unichr_, unicode_
from typing import Dict, List, Tuple, Union
from time import time
import datetime as dt
import csv


TEST_RESULTS = "Stroop_task_test_results_group_21.csv"
TEST_TYPE = 'Button press'
##### CONFIG #####

# Colors abd screen
col_white = (250, 250, 250)
col_black = (0, 0, 0)
col_gray = (220, 220, 220)
col_red = (250, 0, 0)
col_green = (0, 200, 0)
col_blue = (0, 0, 250)
col_yellow = (250,250,0)

BACKGR_COL = col_gray
SCREEN_SIZE = (700, 500)


# Experiment

n_trials = 5

WORDS    = ("red", "green", "blue")

COLORS   = {"red": col_red,
            "green": col_green,
            "blue": col_blue}

KEYS     = {"red": K_b,
            "green": K_n,
            "blue": K_m}



### PYGAME STARTUP ###

pygame.init()
pygame.display.set_mode(SCREEN_SIZE) 
pygame.display.set_caption("Stroop Test")

screen = pygame.display.get_surface()
# screen.fill(BACKGR_COL)

font = pygame.font.Font(None, 80)
font_small = pygame.font.Font(None, 40) 


### MAIN PROGRAM ###

def main():
    
    STATE = "welcome"    
    trial_number = 0    
    
    while True:
        
        # refreshing the surface
        screen.fill(BACKGR_COL)

        # Event loop
        for event in pygame.event.get():
            
            # interactive transitionals                     
            if STATE == "welcome":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "prepare_trial"
                    print(STATE)
                    continue
                    
            if STATE == "trial":
                if event.type == KEYDOWN and event.key in KEYS.values():
                    time_when_reacted = time()
                    this_reaction_time = time_when_reacted - time_when_presented
                    this_correctness = (event.key == KEYS[this_color])
                    if event.key == KEYS['red']:
                        pressed_key = 'red'
                    elif event.key == KEYS['green']:
                        pressed_key = 'green'
                    elif event.key == KEYS['blue']:
                        pressed_key = 'blue'
                    write_csv(TEST_TYPE, trial_number, this_reaction_time, this_correctness, this_color, this_word, pressed_key)
                    STATE = "feedback"
                    print(STATE)
                    continue
    
            if STATE == "feedback":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    if trial_number < n_trials:
                        STATE = "prepare_trial"
                    else:
                        STATE = "goodbye"
                    print(STATE)
                    continue
                    
            
            if event.type == QUIT:
                STATE = "quit"
                print(STATE)
                break

        
        # automatic transitionals
        if STATE == "prepare_trial":
                trial_number = trial_number + 1
                # this_word  = pick_color()
                # this_color = pick_color()
                this_color, this_word = pick_color_and_word()
                time_when_presented = time()
                STATE = "trial"
                print(STATE)    

        
        
        # Presentitionals
        if STATE == "welcome":
            draw_welcome()
            draw_button(SCREEN_SIZE[0]*1/5, 450, "Red: B", col_red)
            draw_button(SCREEN_SIZE[0]*2.5/5, 450, "Green: N", col_green)
            draw_button(SCREEN_SIZE[0]*4/5, 450, "Blue: M", col_blue)
        
        if STATE == "trial":
            draw_stimulus(this_color, this_word)
            draw_button(SCREEN_SIZE[0]*1/5, 450, "Red: B", col_red)
            draw_button(SCREEN_SIZE[0]*2.5/5, 450, "Green: N", col_green)
            draw_button(SCREEN_SIZE[0]*4/5, 450, "Blue: M", col_blue)
        
        if STATE == "feedback":
            draw_feedback(this_correctness, this_reaction_time)
        
        if STATE == "goodbye":
            draw_goodbye()
        
        if STATE == "quit":
            pygame.quit()
            sys.exit()
        
        # Updating the display
        pygame.display.update()
        

# Function definitions
        
def pick_color():
    """ Return a random word.
    """
    random_number = random.randint(0,2)
    return WORDS[random_number]

def draw_button(xpos, ypos, label, color):
    text = font_small.render(label, True, color, BACKGR_COL)
    text_rectangle = text.get_rect()
    text_rectangle.center = (xpos, ypos)
    screen.blit(text, text_rectangle)

def draw_welcome():
    text_surface = font.render("STROOP Experiment", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
    screen.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Press Spacebar to continue", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,300)
    screen.blit(text_surface, text_rectangle)

def draw_stimulus(color, word):
    text_surface = font.render(word, True, COLORS[color], col_gray)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
    screen.blit(text_surface, text_rectangle)

def draw_feedback(correct, reaction_time):
    if correct:
        text_surface = font_small.render("correct", True, col_black, BACKGR_COL)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
        screen.blit(text_surface, text_rectangle)
        text_surface = font_small.render(str(int(reaction_time * 1000)) + "ms", True, col_black, BACKGR_COL)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0]/2.0,200)
        screen.blit(text_surface, text_rectangle)
    else:
        text_surface = font_small.render("Wrong key!", True, col_red, BACKGR_COL)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
        screen.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Press Spacebar to continue", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,300)
    screen.blit(text_surface, text_rectangle)

def draw_goodbye():
    text_surface = font_small.render("END OF THE EXPERIMENT", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
    screen.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Close the application.", True, col_black, BACKGR_COL)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,200)
    screen.blit(text_surface, text_rectangle)


def pick_color_and_word() -> List:
    """
    A color and word are randomly chosen from the WORDS tuple.

    Retrieve the color and word by unpacking them from the return statement.
    color, word = pick_color_and_word()
    """
    return random.sample(WORDS, 2)

def write_csv(*coef):
    """
    Writes the input as a new row in a csv file (called TEST_RESULTS). 
    The first entry of the row is always a timestamp.
    """
    with open(TEST_RESULTS, 'a+', newline='') as file:
        writer = csv.writer(file)
        row = [dt.datetime.utcnow().isoformat()]
        row.extend(coef)
        writer.writerow(row)


main()