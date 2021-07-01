import numpy as np
from numpy import linalg as la
import random
import pygame
import time

# screen dimensions
height = 400
width = 400
canvasx = width / height

# movement

mode = 1
orbit_x = 0
orbit_y = 0
distance = 0
angle_vel = 0.1
lin_vel = 0.01
input_refresh = 5

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



# classes

class Transformation: # transformation that encodes the camera position

    def __init__(self): # initialize as an identity matrix
        self.angle = 0 # saves angle and computes a new transformation for the angle each time to prevent error accumulation
        self.matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    def rotatex(self, theta, set):
        if set == True:
            self.__init__()
        self.angle += theta
        sin = np.sin(self.angle)
        cos = np.cos(self.angle)
        self.matrix = np.array([[1, 0, 0, 0], # the matrix format for rotating about the x-axis
                             [0, cos, sin, 0],
                             [0, -1 * sin, cos, 0],
                             [0, 0, 0, 1]])

    def rotatey(self, theta, set):
        if set == True:
            self.__init__()
        self.angle += theta
        sin = np.sin(self.angle)
        cos = np.cos(self.angle)
        self.matrix = np.array([[cos, 0, -1 * sin, 0],
                             [0, 1, 0, 0],
                             [sin, 0, cos, 0],
                             [0, 0, 0, 1]])

    def rotatez(self, theta, set):
        if set == True:
            self.__init__()
        self.angle += theta
        sin = np.sin(self.angle)
        cos = np.cos(self.angle)
        self.matrix = np.array([[cos, -1 * sin, 0, 0],
                             [sin, cos, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    def translate(self, x, y, z, set):
        if set == True:
            self.__init__()
        self.matrix[0, 3] += x
        self.matrix[1, 3] += y
        self.matrix[2, 3] += z

class Triangle:

    def __init__(self, array):
        self.points = array

    def transform(self, matrix):
        result = np.zeros((3, 3))
        for i in range(3):
            result[i] = np.delete(matrix.dot(np.append(self.points[i], [1])), 3)

        return result

    def project(self):
        result = np.zeros((3, 2))
        for i in range(3):
            for x in range(2):
                if self.points[i][2] > -1:
                    result[0][0] = "nan"
                else:
                    result[i][x] = self.points[i][x]/-self.points[i][2]
        return result

    def rasterize(self, point, vertices):
        cproducts = [0, 0, 0]
        for i in range(3):
            vector1 = np.subtract(point, vertices[i])
            vector2 = np.subtract(vertices[(i+1)%3], vertices[i])
            cproducts[i] = cross(vector1, vector2)
        if abs(sum(cproducts)) == sum(list(map(abs, cproducts))):
            return 1
        else:
            return 0

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
        result = weights[0] * self.points[0][2] + weights[1] * self.points[1][2] + weights[2] * self.points[2][2]
        return result

# functions

def initscreen():
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D")
    screen.fill((0, 0, 0))
    pixels = list(range(height*width))
    random.shuffle(pixels)
    return (screen, pixels)

def draw(screen, pixel, color):
    pygame.draw.circle(screen, color, pixel, 1)

def cross(A, B):
    return (A[0]*B[1]) - (A[1] * B[0])

def input(camera, rotationx, rotationy, rotationz, translation):
    update = 0

    if keyboard.is_pressed('w'): # keyboard inputs modify separate matrices for each axis
        if mode == 'orbit':
            orbit_x += angle_vel
            rotationx.rotatex(orbit_x + np.pi, True)
        rotationx.rotatex(-angle_vel, False)
        update = 1

    if keyboard.is_pressed('s'):
        rotationx.rotatex(angle_vel, False)
        update = 1

    if keyboard.is_pressed('a'):
        rotationy.rotatey(angle_vel, False)
        update = 1

    if keyboard.is_pressed('d'):
        rotationy.rotatey(-angle_vel, False)
        update = 1

    if keyboard.is_pressed('down_arrow'):
        y = 0
        x = np.cos(rotationy.angle + (np.pi / 2)) * lin_vel
        z = np.sin(rotationy.angle + (np.pi / 2)) * lin_vel
        translation.translate(x, y, z)
        update = 1

    if keyboard.is_pressed('up_arrow'):
        y = 0
        x = np.cos(rotationy.angle - (np.pi / 2)) * lin_vel
        z = np.sin(rotationy.angle - (np.pi / 2)) * lin_vel
        translation.translate(x, y, z)
        update = 1

    if keyboard.is_pressed('right_arrow'):
        y = -1 * (np.sin(rotationx.angle) * lin_vel)
        x = -1 * (np.cos(rotationy.angle) * np.cos(rotationx.angle) * lin_vel)
        z = -1 * (np.sin(rotationy.angle) * np.cos(rotationx.angle) * lin_vel)
        translation.translate(x, y, z)
        update = 1

    if keyboard.is_pressed('left_arrow'):
        y = np.sin(rotationx.angle) * lin_vel
        x = np.cos(rotationy.angle) * np.cos(rotationx.angle) * lin_vel
        z = np.sin(rotationy.angle) * np.cos(rotationx.angle) * lin_vel
        translation.translate(x, y, z)
        update = 1

    if update == 1:
        camera.matrix = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        camera.matrix = camera.matrix.dot(translation.matrix)  # matrices are combined
        camera.matrix = camera.matrix.dot(rotationy.matrix)
        camera.matrix = camera.matrix.dot(rotationx.matrix)
        camera.matrix = camera.matrix.dot(rotationz.matrix)

    return update

def render(screen, camtriangles, projtriangles, x, y):
    maxz = 0
    pixel = np.array([((-0.5 * canvasx) - (0.5 * (canvasx / width)) + (x * canvasx / width)),
                      (-0.5 - (0.5 / height) + (y / height))])
    for i in range(triangles.shape[0]):
        triangle = Triangle(camtriangles[i])
        if triangle.rasterize(pixel, projtriangles[i]) == 1:
            weights = triangle.barycenter(pixel, projtriangles[i])
            z = triangle.z(weights)
            if maxz == 0 or z > maxz:
                draw(screen, (x, y), triangles[i][3])
                maxz = z
    if maxz == 0:
        draw(screen, (x, y), (0, 0, 0))
    pygame.display.flip()

def loop(screen, camera, rotationx, rotationy, rotationz, translation, camtriangles, projtriangles, cycle_start, pixels, p):
    if time.perf_counter() - cycle_start > 1/input_refresh:
        #update = input(camera, rotationx, rotationy, rotationz, translation)
        '''if update == 1:
            screen.fill((0, 0, 0))
            p = 0
            inverse = la.inv(camera.matrix)

            camtriangles = np.zeros((triangles.shape[0], triangles.shape[1] - 1, triangles.shape[2]))
            projtriangles = np.zeros((triangles.shape[0], triangles.shape[1] - 1, triangles.shape[2] - 1))

            for i in range(triangles.shape[0]):
                triangle = Triangle(np.stack((triangles[i][0], triangles[i][1], triangles[i][2])))
                camtriangles[i] = triangle.transform(inverse)
                triangle = Triangle(camtriangles[i])
                projtriangles[i] = triangle.project()'''

    if p < width * height:
        pixel = pixels[p]
        x, y = pixel % width, int(pixel/width)
        render(screen, camtriangles, projtriangles, x, y)
        p += 1

    return camtriangles, projtriangles, p

def run(): # main program that renders the saved triangles
    done = False
    rotationx = Transformation() # create the rotation matrices
    rotationy = Transformation()
    rotationz = Transformation()
    translation = Transformation()
    camera1 = Transformation()
    rotationz.rotatez(180 * np.pi / 180, False) # set the initial camera position and angle
    rotationy.rotatey(-20 * np.pi / 180, False)
    rotationx.rotatex(20 * np.pi / 180, False)
    translation.translate(2, 2, 5, False)
    screen, pixels = initscreen()
    camera1.matrix = camera1.matrix.dot(translation.matrix)  # matrices are combined
    camera1.matrix = camera1.matrix.dot(rotationy.matrix)
    camera1.matrix = camera1.matrix.dot(rotationx.matrix)
    camera1.matrix = camera1.matrix.dot(rotationz.matrix)

    p = 0
    inverse = la.inv(camera1.matrix)
    camtriangles = np.zeros((triangles.shape[0], triangles.shape[1] - 1, triangles.shape[2]))
    projtriangles = np.zeros((triangles.shape[0], triangles.shape[1] - 1, triangles.shape[2] - 1))
    for i in range(triangles.shape[0]):
        triangle = Triangle(np.stack((triangles[i][0], triangles[i][1], triangles[i][2])))
        camtriangles[i] = triangle.transform(inverse)
        triangle = Triangle(camtriangles[i])
        projtriangles[i] = triangle.project()

    cycle_start = time.perf_counter()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        timer = time.perf_counter()
        camtriangles, projtriangles, p = loop(screen, camera1, rotationx, rotationy, rotationz, translation, camtriangles, projtriangles, cycle_start, pixels, p)
        print(time.perf_counter() - timer)
        pygame.display.flip()

run()
