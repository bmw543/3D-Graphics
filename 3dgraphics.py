import numpy as np
from numpy import linalg as la
import math
import random
import pygame
import time

# triangles: vertices 1-3, RGB color

triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [255, 0, 0]],
                      [[1, 0, 0], [0, 1, 0], [1, 1, 0], [255, 0, 0]],
                      [[0, 0, 1], [1, 0, 1], [0, 1, 1], [255, 0, 0]],
                      [[1, 0, 1], [0, 1, 1], [1, 1, 1], [255, 0, 0]],
                      [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 255]],
                      [[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 255]],
                      [[0, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 255]],
                      [[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 255]],
                      [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 255, 0]],
                      [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 255, 0]],
                      [[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 255, 0]],
                      [[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 255, 0]]])

# screen dimensions

height = 400
width = 400
canvasx = width / height

# variables

render_distance = 1000

# classes

class Transformation: # transformation that encodes the camera position

    def __init__(self): # initialize as an identity matrix
        self.angles = np.zeros(3)
        self.rotation, self.translation, self.matrix = np.identity(4), np.identity(4), np.identity(4)

    def rotate(self, angles):
        self.angles = angles
        sin_x = np.sin(self.angles[0])
        cos_x = np.cos(self.angles[0])
        sin_y = np.sin(self.angles[1])
        cos_y = np.cos(self.angles[1])
        sin_z = np.sin(self.angles[2])
        cos_z = np.cos(self.angles[2])
        rotation_x = np.array([[1, 0, 0, 0], # the matrix format for rotating about the x-axis
                               [0, cos_x, sin_x, 0],
                               [0, -1 * sin_x, cos_x, 0],
                               [0, 0, 0, 1]])
        rotation_y = np.array([[cos_y, 0, -1 * sin_y, 0],
                               [0, 1, 0, 0],
                               [sin_y, 0, cos_y, 0],
                               [0, 0, 0, 1]])
        rotation_z = np.array([[cos_z, -1 * sin_z, 0, 0],
                               [sin_z, cos_z, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
        self.rotation = rotation_x.dot(rotation_y).dot(rotation_z)
        self.matrix = la.inv(self.translation.dot(self.rotation))

    def translate(self, pos):
        self.translation  = np.array([[1, 0, 0, pos[0]],
                                      [0, 1, 0, pos[1]],
                                      [0, 0, 1, pos[2]],
                                      [0, 0, 0, 1]])
        self.matrix = la.inv(self.rotation.dot(self.translation))

class Triangle:

    def __init__(self, triangle, color):
        self.vertices = triangle
        self.transf_vertices = np.zeros(triangle.shape)
        self.proj_vertices = np.zeros((triangle.shape[0], triangle.shape[1]-2))
        self.color = color

    def transform(self, transformation):
        for i in range(3):
            self.transf_vertices[i] = transformation.matrix.dot(self.vertices[i])

    def project(self):
        for i in range(3):
            if self.transf_vertices[i][2] <= -1:
                self.proj_vertices[i] = np.divide(self.transf_vertices[i][:3][:2],-self.transf_vertices[i][2])
            else:
                return(False)
        return(True)

    def rasterize(self, point, vertices):
        cproducts = [0, 0, 0]
        for i in range(3):
            vector1 = np.subtract(point, vertices[i])
            vector2 = np.subtract(vertices[(i+1)%3], vertices[i])
            cproducts[i] = cross(vector1, vector2)
        if abs(sum(cproducts)) == sum(list(map(abs, cproducts))):
            return True
        else:
            return False

    def barycenter(self, point, vertices):
        coeffs = np.zeros((3, 3))
        deps = np.zeros(3)
        for i in range(2):
            for j in range(3):
                coeffs[i][j] = vertices[j][i]
        coeffs[2] = [1, 1, 1]
        deps[0] = point[0]
        deps[1] = point[1]
        deps[2] = 1
        result = la.solve(coeffs, deps)
        return result

    def z(self, weights):
        result = weights[0] * self.transf_vertices[0][2] + weights[1] * self.transf_vertices[1][2] + weights[2] * self.transf_vertices[2][2]
        return result

    def render(self, screen, transformation):
        self.transform(transformation)
        if self.project():
            min_x = math.floor(width*((np.min(self.proj_vertices[:,0])/canvasx)+(1/2)))
            max_x = math.ceil(width*((np.max(self.proj_vertices[:,0])/canvasx)+(1/2)))
            min_y = math.floor(height*(1*np.min(self.proj_vertices[:,1])+(1/2)))
            max_y = math.ceil(height*(1*np.max(self.proj_vertices[:,1])+(1/2)))
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    pixel = np.array([((-0.5 * canvasx) - (0.5 * (canvasx / width)) + (x * canvasx / width)),
                                      (-0.5 - (0.5 / height) + (y / height))])
                    if self.rasterize(pixel, self.proj_vertices):
                        weights = self.barycenter(pixel, self.proj_vertices)
                        z = self.z(weights)
                        if z > zmap[x][y]:
                            draw(screen, (x, y), self.color)
                            zmap[x][y] = z
            pygame.display.flip()

# functions

def initscreen():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D")
    screen.fill((0, 0, 0))
    return (screen)

def draw(screen, pixel, color):
    pygame.draw.circle(screen, color, pixel, 1)

def cross(A, B):
    return (A[0]*B[1]) - (A[1] * B[0])

def run():
    colors = triangles[:,3]
    vertices = np.insert(triangles[:,[0,1,2]], 3, 1, axis=2)
    global zmap
    zmap = np.full((width, height), float(-1*render_distance))

    camera = Transformation()
    camera.translate((2, 2, 5))
    camera.rotate((20 * np.pi / 180, -20 * np.pi / 180, 180 * np.pi / 180))

    screen = initscreen()

    while(True):
        for i in range(triangles.shape[0]):
            print(i)
            triangle = Triangle(vertices[i], triangles[i][3])
            #print(camera.matrix)
            #print(triangle.vertices)
            triangle.render(screen, camera)
run()
