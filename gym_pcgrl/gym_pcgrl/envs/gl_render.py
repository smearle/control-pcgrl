from pdb import set_trace as TT

import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

pos_x, pos_y, rot_x, rot_y, zoom = 0, 0, 0, 0, -0.5

def init_display():
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)  
        glTranslate(0.0,-5,-10)
        glEnable(GL_DEPTH_TEST)
    
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        display = display

        return display

def render_opengl(display, rep_map, paths=[]):
    global rot_x, rot_y, zoom
    i = 0
    width, height, depth = rep_map.shape
    # while True:
    for _ in range(1):

        keys_pressed = pygame.mouse.get_pressed()


        # if keys_pressed[pygame.K_w]:
        #     glTranslatef(0.0, -0.1, 0.0)

        # if keys_pressed[pygame.K_s]:
        #     glTranslatef(0.0, 0.1, 0.0)

        # if keys_pressed[pygame.K_d]:
        #     glTranslatef(-0.1, 0.0, 0.0)

        # if keys_pressed[pygame.K_a]:
        #     glTranslatef(0.1, 0.0, 0.0)

        # if keys_pressed[pygame.K_q]:
        #     glRotatef(1, 0, 1, 0)
        # if keys_pressed[pygame.K_e]:
        #     print('q')
        #     glRotatef(-1, 0, 1, 0)

        l_button_down = keys_pressed[0] == 1
        r_button_down = keys_pressed[2] == 1
        global pos_x, pos_y, rot_x, rot_y, zoom

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                busy = False
            elif event.type == pygame.MOUSEMOTION:
                if l_button_down:
                    rot_x += event.rel[1]
                    rot_y += event.rel[0]
                elif r_button_down:
                    pos_x += event.rel[0] * 0.1
                    pos_y += event.rel[1] * 0.1 
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    zoom += 0.2
                if event.button == 5:
                    zoom -= 0.2


            # control_check()
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         glTranslatef(-0.5, 0.0, 0.0)
            #     if event.key == pygame.K_RIGHT:
            #         glTranslatef(0.5, 0.0, 0.0)
            #     if event.key == pygame.K_UP:
            #         glTranslatef(0.0, 1.0, 0.0)
            #     if event.key == pygame.K_DOWN:
            #         glTranslatef(0.0, -1.0, 0.0)

            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     if event.button == 4:
            #         glTranslatef(0.0, 0.0, 1.0)
            #     if event.button == 5:
            #         glTranslatef(0.0, 0.0, -1.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glTranslatef(pos_x, pos_y, zoom)
        glRotatef(rot_x, 1, 0, 0)    
        glRotatef(rot_y, 0, 1, 0)    

        Plane()
        dirt_cubes = np.argwhere(rep_map == 1)
        for (y, x, z) in dirt_cubes:
            x, z = x - width/2, z - depth/2
            Cube(loc=(x,y,z), color=(.89, .44, .18))

        for path in paths:
            for (z, x, y) in path:
                x, z = x - width/2, z - depth/2
                Cube(loc=(x,y,z), color=(.44, .89, .18))

        glPopMatrix()
        pygame.display.flip()
        pygame.time.wait(1)
        i += 1
        # pygame.time.wait(10)

# from sentdex youtube video
def control_check(keys_pressed):
    button_down = keys_pressed[0] == 1
    global rot_x, rot_y, zoom

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            busy = False
        elif event.type == pygame.MOUSEMOTION:
            if button_down:
                rot_x += event.rel[1]
                rot_y += event.rel[0]
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                zoom += 0.2
            if event.button == 5:
                zoom -= 0.2

    if keys_pressed[pygame.K_w]:
        glTranslatef(0.0, -0.1, 0.0)

    if keys_pressed[pygame.K_s]:
        glTranslatef(0.0, 0.1, 0.0)

    if keys_pressed[pygame.K_d]:
        glTranslatef(-0.1, 0.0, 0.0)

    if keys_pressed[pygame.K_a]:
        glTranslatef(0.1, 0.0, 0.0)

    if keys_pressed[pygame.K_q]:
        print('q')
        glRotatef(1, 0, 1, 0)
    if keys_pressed[pygame.K_e]:
        print('q')
        glRotatef(-1, 0, 1, 0)



ground = (
    (-100,  -1.50, -100),
    ( 100,  -1.50, -100),
    ( 100,  -1.50, 3100),
    (-100,  -1.50, 3100),
)
ground = np.array(ground, dtype=np.float32)
ground_edges = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
)
ground_surface = (
    (0, 1, 2, 3),
)

def Plane(loc=(0, 0, 0), color=(.4, .4, .4)):
    color = np.array(color, dtype=np.float)
    loc = np.array(loc, dtype=np.float)
    glBegin(GL_QUADS)
    for surface in ground_surface:
        x = 0
        for vertex in surface:
            # x+=1
            # glColor3fv(colors[x])
            glColor3fv(np.array(color) + x * np.array([-0.2,-0.2,-0.2]))
            glVertex3fv(ground[vertex] / 2 + loc)
    glEnd()


vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
)
cube_vertices = np.array(vertices)

edges = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3), 
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7),
)

surfaces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6),
)

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
)

def Cube(loc=(0, 0, 0), color=(0, 1, 1)):
    color = np.array(color, dtype=np.float) 
    loc = np.array(loc, dtype=np.float)
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            # glColor3fv(colors[x])
            # glColor3fv(np.array(color) + x * np.array([-0.1,-0.1,-0.1]))
            glColor4fv(np.array((*color, 0.4)) + x * np.array([-0.1,-0.1,-0.1, 0.0]))
            glVertex3fv(cube_vertices[vertex] / 2 + loc)
    glEnd()


def main():
    pygame.init()
    screen = (400, 300)
    pygame.display.set_mode(screen, DOUBLEBUF|OPENGL)

    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (screen[0]/screen[1]), 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)  
    glTranslate(0.0,0.0,-5)


    clock = pygame.time.Clock()
    busy = True
    while busy:

        mouse_buttons = pygame.mouse.get_pressed()
        button_down = mouse_buttons[0] == 1
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                busy = False
            elif event.type == pygame.MOUSEMOTION:
                if button_down:
                    rot_x += event.rel[1]
                    rot_y += event.rel[0]
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    zoom += 0.2
                if event.button == 5:
                    zoom -= 0.2
                
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        glTranslatef(0.0,0.0, zoom)
        glRotatef(rot_x, 1, 0, 0)    
        glRotatef(rot_y, 0, 1, 0)    
        Cube()
        glPopMatrix()
        
        pygame.display.flip()
        clock.tick(100)

    pygame.quit()

if __name__ == "__main__":
    main()