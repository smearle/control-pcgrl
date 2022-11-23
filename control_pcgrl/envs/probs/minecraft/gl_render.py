from pdb import set_trace as TT

import numpy as np
import pygame
from pygame.locals import *
import torch as th
from torch import nn

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

AIR = 0
DIRT = 1
PATH = 2
ENTRANCE = 3
EXIT = 4

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 1, 1),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
)

pos_x, pos_y, rot_x, rot_y, zoom = 0, 0, 0, 0, -0.5

adjs = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
adjs += 1

cube_vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1),
)
cube_vertices = np.array(cube_vertices)

cube_edges = (
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

cube_surfaces = (
    (4, 0, 3, 6),  # bottom
    (4, 5, 1, 0),  # right
    (1, 5, 7, 2),  # top
    (3, 2, 7, 6),  # left
    (6, 7, 5, 4),  # front
    (0, 1, 2, 3),  # back
)

class CubeFaceNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 6, kernel_size=3, padding=1, bias=False)
        # Hand-code the weights. Does not need gradient.
        self.conv.weight = nn.Parameter(th.zeros_like(self.conv.weight), requires_grad=False)
        # Activate an adjacency channel if the current cube is active and the adjacent cube is not.
        self.conv.weight[:, 0, 1, 1, 1] = 1
        for i, adj in enumerate(adjs):
            self.conv.weight[i, 0, adj[0], adj[1], adj[2]] = -1

    def forward(self, x):
        return th.relu(self.conv(x))


cube_face_nn = CubeFaceNN()


class Scene():
    def __init__(self):
        self.i = 0
        pygame.init()
        display = (1280, 720)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

        glMatrixMode(GL_MODELVIEW)  
        glTranslate(0.0,-5,-20)
        glEnable(GL_DEPTH_TEST)
    
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        self.display = display


    def render(self, rep_map, paths=[], bordered=False):
        display = self.display
        rep_map = rep_map.copy()
        if bordered:
            borders = rep_map.copy()
            borders[1:-1, 1:-1, 1:-1] = DIRT
            ent_exit = np.where(borders == AIR)
            rep_map[:, :, 0] = rep_map[:, :, -1] = AIR
            rep_map[:, 0, :] = rep_map[:, -1, :] = AIR
            rep_map[0, :, :] = rep_map[-1, :, :] = AIR
            rep_map[ent_exit] = ENTRANCE
        global rot_x, rot_y, zoom
        width, height, depth = rep_map.shape
        # while True:
        for _ in range(1):

            keys_pressed = pygame.mouse.get_pressed()
            l_button_down = keys_pressed[0] == 1
            r_button_down = keys_pressed[2] == 1
            global pos_x, pos_y, rot_x, rot_y, zoom

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    busy = False
                elif event.type == pygame.MOUSEMOTION:
                    if l_button_down:
                        rot_x += event.rel[1] * .5
                        rot_y += event.rel[0] * .5
                    if r_button_down:
                        pos_x += event.rel[0] * 0.05
                        pos_y -= event.rel[1] * 0.05 
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        zoom += 0.2
                    if event.button == 5:
                        zoom -= 0.2

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glPushMatrix()
            glTranslatef(pos_x, pos_y, zoom)
            glRotatef(rot_x, 1, 0, 0)    
            glRotatef(rot_y, 0, 1, 0)    

            # The draw order affects the transparency of the objects in the scene.
            for path in paths:
                for (z, x, y) in path:
                    x, z = x - width/2, z - depth/2
                    Cube(loc=(x,y,z), color=(.44, .89, .18, 0.4))

            ent_cubes = np.argwhere(rep_map == ENTRANCE)
            for (y, x, z) in ent_cubes:
                x, z = x - width/2, z - depth/2
                Cube(loc=(x,y,z), color=(.12, .1, .80, 0.4))

            dirt_cubes = th.Tensor((rep_map == DIRT))[None, ...]
            dirt_cube_faces = cube_face_nn(dirt_cubes)
            # print(f'{dirt_cubes.sum()} dirt cubes, {dirt_cube_faces.sum()} dirt cube faces')
            cube_color = (.89, .44, .18, 0.3)
            for (f, y, x, z) in th.argwhere(dirt_cube_faces > .5):
                x, z = x - width/2, z - depth/2
                glBegin(GL_QUADS)
                i = 0
                for vertex in cube_surfaces[f]:
                    i +=1
                    glColor4fv(cube_color + (i * np.array([-0.1,-0.1,-0.1, 0.0])))
                    glVertex3fv(cube_vertices[vertex] / 2 + np.array([x, y, z]))
                glEnd()

            # dirt_cubes = np.where(rep_map == DIRT)
            # dirt_cubes = np.argwhere(rep_map == DIRT)
            # dirt_cube_faces[dirt_cube_faces]
            # for (y, x, z) in dirt_cubes:
                # x, z = x - width/2, z - depth/2
                # color = np.array([.89, .44, .18, 0.3])
                # Cube(loc=(x,y,z), color=(.89, .44, .18, 0.3))
                # glBegin(GL_QUADS)
                # for surface in surfaces:
                #     x = 0
                #     for vertex in surface:
                #         x+=1
                #         glColor4fv(color + (x * np.array([-0.1,-0.1,-0.1, 0.0])))
                #         glVertex3fv(cube_vertices[vertex] / 2 + loc)
            # glEnd()
            Plane((0,-.5,0))

            glPopMatrix()
            pygame.display.flip()
            pygame.time.wait(1)
            self.i += 1
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
    (-100,  0, -100),
    ( 100,  0, -100),
    ( 100,  0, 3100),
    (-100,  0, 3100),
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

def Plane(loc=(0, 0, 0), color=(.2, .2, .20, 1)):
    color = np.array(color, dtype=np.float)
    loc = np.array(loc, dtype=np.float)
    glBegin(GL_QUADS)
    for surface in ground_surface:
        x = 0
        for vertex in surface:
            # x+=1
            # glColor3fv(colors[x])
            glColor4fv(np.array(color) + x * np.array([-0.2,-0.2,-0.2,0]))
            glVertex3fv(ground[vertex] + loc)
    glEnd()


def Cube(loc=(0, 0, 0), color=(0, 1, 1, 0.2)):
    color = np.array(color, dtype=np.float) 
    loc = np.array(loc, dtype=np.float)
    glBegin(GL_QUADS)
    for surface in cube_surfaces:
        x = 0
        for vertex in surface:
            x+=1
            # glColor3fv(colors[x])
            # glColor3fv(np.array(color) + x * np.array([-0.1,-0.1,-0.1]))
            glColor4fv(color + (x * np.array([-0.1,-0.1,-0.1, 0.0])))
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