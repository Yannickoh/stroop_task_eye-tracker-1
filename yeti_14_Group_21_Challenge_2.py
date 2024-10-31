## YETI 14: Collecting 2-dim calibration data for quadrant SBG
## input = calibration point table
## Results = table with target coordinates and quadrant brightness
## Stroop task added to YETI 14
## Stroop task performed with eyetracking
## Stroop task brought back to two options
## Collected data from stroop task saved in CSV file


from typing import Dict, List, Tuple, Union
from time import time
import datetime as dt

YETI = 14
YETI_NAME = "Yeti" + str(YETI)
TITLE = "Multi-point calibration measures with quadrant brightness"
AUTHOR = "M SCHMETTOW"
CONFIG = "yeti_14_group_21.xlsx"
RESULTS = "yeti_14_group_21" + str(dt.datetime.utcnow().timestamp()) + ".xlsx"
TEST_RESULTS = "Stroop_task_test_results_group_21.csv"
TEST_TYPE = 'Eye tracking'

import sys
import os
import logging as log

# DS
import random
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
import csv
# CV
import cv2 as cv
# PG
import pygame as pg
from pygame.locals import *


##### Preparations #####
# CV
log.basicConfig(filename='YET.log',level=log.INFO)
# adjust to the correct camera here:
YET = cv.VideoCapture(0)
if YET.isOpened():
    width = int(YET.get(cv.CAP_PROP_FRAME_WIDTH ))
    height = int(YET.get(cv.CAP_PROP_FRAME_HEIGHT ))
    fps =  YET.get(cv.CAP_PROP_FPS)
    dim = (width, height)
else:
    print('Unable to load camera.')
    exit()


# Reading the CV model for eye detection
eyeCascadeFile = "haarcascade_eye.xml"
if os.path.isfile(eyeCascadeFile):
    eyeCascade = cv.CascadeClassifier(eyeCascadeFile)
else:
    sys.exit(eyeCascadeFile + ' not found. CWD: ' + os.getcwd()) 


# Reading calibration point matrix from Excel table
Targets = pd.read_excel(CONFIG)


# Color codes
col_black = (0, 0, 0)
col_gray = (120, 120, 120)
col_light_gray = (160, 160, 160)
col_red = (250, 0, 0)
col_green = (0, 255, 0)
col_blue = (0, 0, 250)
col_yellow = (250, 250, 0)
col_white = (255, 255, 255)
col_orange = (255,120,0)

col_red_dim = (120, 0, 0)
col_green_dim = (0, 60, 0)
col_yellow_dim = (120,120,0)


# width and height in pixel of the screen
SCREEN_W = 800
SCREEN_H = 800
SCREEN_SIZE = (SCREEN_W, SCREEN_H)
CAMERA_SCREEN_W = 200
CAMERA_SCREEN_H = 200
CAMERA_SCREEN_SIZE = (CAMERA_SCREEN_W, CAMERA_SCREEN_H)
CAMERA_SCREEN_LOCATION = ((SCREEN_W -CAMERA_SCREEN_W)/2, (SCREEN_H -CAMERA_SCREEN_H)/2)


pg.init()
pg.display.set_mode(SCREEN_SIZE)
pg.display.set_caption(YETI_NAME)
FONT = pg.font.Font('freesansbold.ttf',24)

SCREEN = pg.display.get_surface()
font = pg.font.Font(None, 60)
font_small = pg.font.Font(None, 40) 


# Experiment
n_trials = 5

## number of words/colors must always be more or equal to the amount of targets ##
WORDS = ("red", "green", "blue")
COLORS = {
    "red": col_red,
    "green": col_green,
    "blue": col_blue
}
# KEYS = {
#     "red": K_b,
#     "green": K_n,
#     "blue": K_m
# }
BUTTONS_Y_POS = 550
CAPTURE_BOX_Y_POS = 675
CAPTURE_BOX_WIDTH = 150
CAPTURE_BOX_HEIGHT = 150
CAPTURE_TARGETS = {
    "left": {
        "x": SCREEN_SIZE[0]*1/5,
        "y": CAPTURE_BOX_Y_POS,
        "width": CAPTURE_BOX_WIDTH,
        "height": CAPTURE_BOX_HEIGHT,
        "color": "left"
    },
    ## Possible third/middle target ##
    # "middle": {
    #     "x": SCREEN_SIZE[0]*2.5/5,
    #     "y": CAPTURE_BOX_Y_POS,
    #     "width": CAPTURE_BOX_WIDTH,
    #     "height": CAPTURE_BOX_HEIGHT,
    #     "color": "middle"
    # },
    "right": {
        "x": SCREEN_SIZE[0]*4/5,
        "y": CAPTURE_BOX_Y_POS,
        "width": CAPTURE_BOX_WIDTH,
        "height": CAPTURE_BOX_HEIGHT,
        "color": "right"
    }
}
RESET_TARGET = {
    "x": SCREEN_SIZE[0]/2.0,
    "y": 125,
    "width": SCREEN_SIZE[0],
    "height": 250
}

def main():

    ## Initial State
    STATE = "Welcome" # Measure, Target
    DETECTED = False
    TEXT_COL=col_white
    BACKGR_COL=col_black
    
    n_targets = len(Targets)
    active_target = 0
    run = 0
    H_offset, V_offset = (0,0)
    this_pos = (0,0)

    Eyes = []
    OBS_cols = ("Obs", "run", "NW", "NE", "SW", "SE", "x", "y")
    OBS = np.empty(shape = (0, len(OBS_cols)))
    obs = 0

    trial_number = 0  


    ## FAST LOOP
    while True:
        # General frame processing
        ret, Frame = YET.read(1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        F_gray = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)
        # Eye detection. Eye frame is being locked.
        if STATE == "Detect":
            Eyes = eyeCascade.detectMultiScale(
                    F_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100))
            if len(Eyes) > 0:
                DETECTED = True
                x_eye, y_eye, w_eye, h_eye = Eyes[0]
                F_eye = F_gray[y_eye:y_eye+h_eye,x_eye:x_eye+w_eye]
            else: 
                DETECTED = False
                # F_eye = F_gray
                # w_eye, h_eye = (width, height)
        elif STATE != "Welcome":
            F_eye = F_gray[y_eye:y_eye+h_eye,x_eye:x_eye+w_eye]
        if STATE == "Validate" or STATE == "trial" or STATE == "feedback":
            this_quad = np.array(quad_bright(F_eye))
            this_quad.shape = (1,4)
            this_pos = M_0.predict(this_quad)[0,:]
            # print(this_pos)
        if STATE == "trial":
            for target in CAPTURE_TARGETS.keys():
                x_min, x_max, y_min, y_max = calculate_target_boundries(CAPTURE_TARGETS[target])
                if check_capture_event(x_min, x_max, y_min, y_max, np.round(this_pos[0]), np.round(this_pos[1])):
                    time_when_reacted = time()
                    this_reaction_time = time_when_reacted - time_when_presented
                    print(f"Captured {CAPTURE_TARGETS[target]['color']}")
                    ## To debug use the following 3 lines: ##
                    # print("Debug ========")
                    # print(f"This color: {this_color}, This word: {this_word}")
                    # print(CAPTURE_TARGETS)
                    this_correctness = CAPTURE_TARGETS[target]["color"] == this_color
                    write_csv(TEST_TYPE, trial_number, this_reaction_time, this_correctness, this_color, this_word, CAPTURE_TARGETS[target]["color"])
                    STATE = "feedback"
                    print(STATE)
        elif STATE == "feedback":
            x_min, x_max, y_min, y_max = calculate_target_boundries(RESET_TARGET)
            if check_capture_event(x_min, x_max, y_min, y_max, np.round(this_pos[0]), np.round(this_pos[1])):
                if trial_number < n_trials:
                    STATE = "prepare_trial"
                else:
                    STATE = "goodbye"
                print(STATE)


        ## Event handling
        for event in pg.event.get():
            # Interactive transition conditionals (ITC)
            if STATE == "Welcome":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Detect"
                elif event.type == KEYDOWN and event.key == K_b:
                    BACKGR_COL, TEXT_COL = TEXT_COL, BACKGR_COL
            if STATE == "Detect":
                if DETECTED:
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        this_Eye = Eyes[0]
                        x_eye, y_eye, w_eye, h_eye = this_Eye
                        STATE = "Target"
                        print(STATE  + str(active_target))
                        
            elif STATE == "Target":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Measure"
                    print(STATE  + str(active_target))
            elif STATE == "Save":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "Train" 
                if event.type == KEYDOWN and event.key == K_RETURN:
                    # STATE = "Save" 
                    OBS = pd.DataFrame(OBS, columns = OBS_cols)
                    with pd.ExcelWriter(RESULTS) as writer:
                        print(OBS)
                        OBS.to_excel(writer, sheet_name="Obs_" + str(time()), index = False)
                        print(RESULTS)
                if event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target" 
                    active_target = 0 # reset
                    run = run + 1
            elif STATE == "Validate":
                if event.type == KEYDOWN and event.key == K_BACKSPACE:
                    STATE = "Target" 
                    active_target = 0 # reset
                    run += 1
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "stroop_welcome"
            elif STATE == "stroop_welcome":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    STATE = "prepare_trial"
                    print(STATE)
            elif STATE == "feedback":
                if event.type == KEYDOWN and event.key == K_SPACE:
                    if trial_number < n_trials:
                        STATE = "prepare_trial"
                    else:
                        STATE = "goodbye"
                    print(STATE)            
            if event.type == QUIT:
                STATE = "quit"
        if event.type == QUIT:
            YET.release()
            pg.quit()
            sys.exit()


        # Automatic transitionals
        if STATE == "Measure":
            obs = obs + 1
            this_id = (obs, run)
            this_targ = tuple(Targets.to_numpy()[active_target][0:2])
            this_bright = quad_bright(F_eye)
            #print(this_targ.shape, this_id.shape, this_bright.shape)
            this_obs = this_id + this_bright + this_targ
            print(this_obs)
            OBS = np.vstack((OBS, this_obs))
            if (active_target + 1) < n_targets:
                active_target = active_target + 1
                STATE = "Target"
                print(STATE  + str(active_target))
            else:
                print(OBS)
                STATE = "Save"        
        if STATE == "Train":
            M_0 = train_QBG(OBS)
            STATE = "Validate"
        if STATE == "prepare_trial":
            trial_number = trial_number + 1
            ## When there is 2 targets: ##
            this_color, this_word = pick_color_and_word()
            ## When there is 3 targets: ##
            # this_color, this_word, third_color = pick_color_and_word()
            #
            time_when_presented = time()
            ## When there is 2 targets: ##
            CAPTURE_TARGETS["left"]["color"], CAPTURE_TARGETS["right"]["color"] = randomize_targets((this_word, this_color))
            ## When there is 3 targets: ##
            # CAPTURE_TARGETS["left"]["color"], CAPTURE_TARGETS["middle"]["color"], CAPTURE_TARGETS["right"]["color"] = randomize_targets((this_word, this_color, third_color))
            #
            STATE = "trial"
            print(STATE)  


        # Presentitionals
        # over-paint previous with background xcolor
        pg.display.get_surface().fill(BACKGR_COL)
        if STATE == "Welcome":
            draw_text("Welcome, Press Space to continue, b to toggle text mode", (30, 700), TEXT_COL)
        elif STATE == "Detect":
            if DETECTED:
                Img = frame_to_surf(F_eye, CAMERA_SCREEN_SIZE)
            else:
                Img = frame_to_surf(Frame, CAMERA_SCREEN_SIZE)
            SCREEN.blit(Img, CAMERA_SCREEN_LOCATION)
        elif STATE == "Target":
            if DETECTED:
                drawTargets(SCREEN, Targets, active_target)
        elif STATE == "Save":
            draw_text("Press Enter to save calibration data", (30, 600), TEXT_COL)
            draw_text("Press Backspace recalibrate", (30, 650), TEXT_COL)
            draw_text("Press Space to continue", (30, 700), TEXT_COL)
        elif STATE == "Validate":
            draw_text("Check if your eye is being tracked correctly.", (30, 600), TEXT_COL)
            draw_text("Press Space to continue, Backspace to recalibrate.", (30, 700), TEXT_COL)
            draw_rect(this_pos[0] + H_offset - 1, 0, 2, SCREEN_H, stroke_size=1, color = col_green)
            draw_rect(0, this_pos[1] + V_offset - 1, SCREEN_W, 2, stroke_size=1, color = col_green)
            # diagnostics
            draw_text("HPOS: " + str(np.round(this_pos[0])), (510, 250), color=col_green)
            draw_text("VPOS: " + str(np.round(this_pos[1])), (510, 300), color=col_green)
            draw_text("XOFF: " + str(H_offset), (510, 350), color=col_green)
            draw_text("YOFF: " + str(V_offset), (510, 400), color=col_green)

        if STATE == "stroop_welcome":
            draw_welcome(TEXT_COL, BACKGR_COL)
            ## Show target placement used during testing on welcome screen: ##
            # draw_button(SCREEN_SIZE[0]*1/5, BUTTONS_Y_POS, "Left", TEXT_COL, BACKGR_COL)
            # draw_button(SCREEN_SIZE[0]*2.5/5, BUTTONS_Y_POS, "Middle", TEXT_COL, BACKGR_COL)
            # draw_button(SCREEN_SIZE[0]*4/5, BUTTONS_Y_POS, "Right", TEXT_COL, BACKGR_COL)
        elif STATE == "trial":
            draw_stimulus(this_color, this_word, BACKGR_COL)            
            draw_button(SCREEN_SIZE[0]*1/5, BUTTONS_Y_POS, CAPTURE_TARGETS["left"]["color"], TEXT_COL, BACKGR_COL)
            ## Option for a middle/third color ##
            # draw_button(SCREEN_SIZE[0]*2.5/5, BUTTONS_Y_POS, CAPTURE_TARGETS["middle"]["color"], TEXT_COL, BACKGR_COL)
            draw_button(SCREEN_SIZE[0]*4/5, BUTTONS_Y_POS, CAPTURE_TARGETS["right"]["color"], TEXT_COL, BACKGR_COL)
            draw_target_rect(CAPTURE_TARGETS["left"]["x"], CAPTURE_TARGETS["left"]["y"], CAPTURE_TARGETS["left"]["width"], CAPTURE_TARGETS["left"]["height"], TEXT_COL)
            ## Option for a middle/third target ##
            # draw_target_rect(CAPTURE_TARGETS["middle"]["x"], CAPTURE_TARGETS["middle"]["y"], CAPTURE_TARGETS["middle"]["width"], CAPTURE_TARGETS["middle"]["height"], TEXT_COL)
            draw_target_rect(CAPTURE_TARGETS["right"]["x"], CAPTURE_TARGETS["right"]["y"], CAPTURE_TARGETS["right"]["width"], CAPTURE_TARGETS["right"]["height"], TEXT_COL)
            draw_rect(this_pos[0] + H_offset - 1, 0, 2, SCREEN_H, stroke_size=1, color = col_green)
            draw_rect(0, this_pos[1] + V_offset - 1, SCREEN_W, 2, stroke_size=1, color = col_green)
        elif STATE == "feedback":
            draw_feedback(this_correctness, this_reaction_time, TEXT_COL, BACKGR_COL)
            draw_target_rect(RESET_TARGET["x"], RESET_TARGET["y"], RESET_TARGET["width"], RESET_TARGET["height"], TEXT_COL)
            draw_rect(this_pos[0] + H_offset - 1, 0, 2, SCREEN_H, stroke_size=1, color = col_green)
            draw_rect(0, this_pos[1] + V_offset - 1, SCREEN_W, 2, stroke_size=1, color = col_green)
        elif STATE == "goodbye":
            draw_goodbye(TEXT_COL, BACKGR_COL)
        elif STATE == "quit":
            pg.quit()
            sys.exit()
        # update the screen to display the changes you made
        pg.display.update()


def drawTargets(screen, Targets, active = 0):
    # Moved outside loop because it is essentially constant
    radius = 20
    stroke = 10
    for index, target in Targets.iterrows():
        color = col_orange if index == active else col_light_gray
        pos = (target['x'], target['y'])
        pg.draw.circle(screen, color, pos, radius, stroke)

## Converts a cv framebuffer into Pygame image (surface!)
def frame_to_surf(frame, dim):
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB) # convert BGR (cv) to RGB (Pygame)
    img = np.rot90(img) # rotate coordinate system
    surf = pg.surfarray.make_surface(img)
    surf = pg.transform.smoothscale(surf, dim)
    return surf


def draw_text(text, dim, color = (255, 255, 255), center = False):
    x, y = dim
    rendered_text = FONT.render(text, True, color)
    # retrieving the abstract rectangle of the text box
    box = rendered_text.get_rect()
    # this sets the x and y coordinates
    if center:
        box.center = (x,y)
    else:
        box.topleft = (x,y)
    # This puts a pre-rendered object to the screen
    SCREEN.blit(rendered_text, box)

def draw_rect(x, y, width, height,color = (255,255,255), stroke_size = 1):
    """
    A rectangle is drawn with the left top corner coordinate as an input coordinate.
    """
    pg.draw.rect(SCREEN, color, (x, y, width, height), stroke_size)

def draw_target_rect(x, y, width, height, color = (255,255,255), stroke_size = 4):
    """
    A (target) rectangle is drawn with the centre coordinate as an input coordinate.
    """
    pg.draw.rect(SCREEN, color, (x-width/2, y-height/2, width, height), stroke_size)


def quad_bright(frame):
    w, h = np.shape(frame)
    b_NW =  np.mean(frame[0:int(h/2), 0:int(w/2)])
    b_NE =  np.mean(frame[int(h/2):h, 0:int(w/2)])
    b_SW =  np.mean(frame[0:int(h/2), int(w/2):w])
    b_SE =  np.mean(frame[int(h/2):h, int(w/2):w])
    out = (b_NW, b_NE, b_SW, b_SE)
    return(out)

def train_QBG(Obs):
    Quad = Obs[:,2:6]
    Pos = Obs[:,6:8]
    model = lm.LinearRegression()
    model.fit(Quad, Pos)
    return model


# Predicts position based on quad-split
def predict_pos(data, model):
    predictions = model.predict(data)
    return predictions

def draw_button(xpos, ypos, label, color, backgr_col):
    text = font_small.render(label, True, color, backgr_col)
    text_rectangle = text.get_rect()
    text_rectangle.center = (xpos, ypos)
    SCREEN.blit(text, text_rectangle)

def draw_welcome(text_col, backgr_col):
    text_surface = font.render("STROOP Experiment", True, text_col, backgr_col)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Press Spacebar to continue", True, text_col, backgr_col)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,300)
    SCREEN.blit(text_surface, text_rectangle)

def draw_stimulus(color, word, backgr_col):
    text_surface = font.render(word, True, COLORS[color], backgr_col)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
    SCREEN.blit(text_surface, text_rectangle)

def draw_feedback(correct, reaction_time, text_col, backgr_col):
    if correct:
        text_surface = font_small.render("correct", True, text_col, backgr_col)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0]/2.0,100)
        SCREEN.blit(text_surface, text_rectangle)
        text_surface = font_small.render(str(int(reaction_time * 1000)) + "ms", True, text_col, backgr_col)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0]/2.0,125)
        SCREEN.blit(text_surface, text_rectangle)
    else:
        text_surface = font_small.render("Wrong key!", True, text_col, backgr_col)
        text_rectangle = text_surface.get_rect()
        text_rectangle.center = (SCREEN_SIZE[0]/2.0,100)
        SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Press Spacebar to continue", True, text_col, backgr_col)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,350)
    SCREEN.blit(text_surface, text_rectangle)

def draw_goodbye(text_col, backgr_col):
    text_surface = font_small.render("END OF THE EXPERIMENT", True, text_col, backgr_col)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,150)
    SCREEN.blit(text_surface, text_rectangle)
    text_surface = font_small.render("Close the application.", True, text_col, backgr_col)
    text_rectangle = text_surface.get_rect()
    text_rectangle.center = (SCREEN_SIZE[0]/2.0,200)
    SCREEN.blit(text_surface, text_rectangle)


def check_capture_event(x_min, x_max, y_min, y_max, x_pos, y_pos) -> bool:
    """
    A check to see if the current eye position is within a targets boundaries.

    This is done by checking if x and y position are between the minimum and maximum x and y values of a target.
    """
    return x_min < x_pos < x_max and y_min < y_pos < y_max

def calculate_target_boundries(target: Dict) -> Tuple:
    """
    The x and y boundary values of a target (square) are calculated based on the targets 

    x_min, x_max, y_min, y_max = calculate_target_boundries(x, y, width, height)
    """
    x_min = target["x"] - target["width"]/2
    x_max = target["x"] + target["width"]/2
    y_min = target["y"] - target["height"]/2
    y_max = target["y"] + target["height"]/2
    return x_min, x_max, y_min, y_max

def pick_color_and_word() -> List:
    """
    A color and word are randomly chosen from the WORDS tuple.

    Retrieve the color and word by unpacking them from the return statement.
    color, word = pick_color_and_word()
    """
    return random.sample(WORDS, len(CAPTURE_TARGETS))

def randomize_targets(targets: Tuple) -> List:
    """
    Returns the tuple in a randomised order.

    This is used to randomly assign this_word and this_color to the left and right target.
    left, right = randomize_targets(targets)
    """
    l = list(targets)
    random.shuffle(l)
    return l

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

if __name__ == "__main__":
    main()
