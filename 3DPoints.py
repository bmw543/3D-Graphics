import time
import keyboard
import numpy as np
import pygame

# screen dimensions
HEIGHT = 794
WIDTH = 1440
canvasx = (WIDTH / HEIGHT)

# colors
white = (255, 255, 255)
black = (0, 0, 0)

# Movement
maxSpeed = 0.05

# points
points = np.array([[0, 1, 0],
                   [1, -1, 1],
                   [1, -1, -1],
                   [-1, -1, 1],
                   [-1, -1, -1]])


class Transformation:

    # initializes as identity matrix
    def __init__(self, dimensions):
        self.angle = 0
        self.matrix = np.zeros([dimensions, dimensions])
        self.dimensions = dimensions
        for x in range(dimensions):
            self.matrix[x, x] = 1

    # transformation functions
    def rotate(self, direction, theta):
        sin = np.sin(theta)
        cos = np.cos(theta)
        if direction == 'x':
            self.angle += theta
            rotation = np.array([[1, 0, 0, 0],
                                 [0, cos, sin, 0],
                                 [0, -1 * sin, cos, 0],
                                 [0, 0, 0, 1]])
        if direction == 'y':
            self.angle += theta
            rotation = np.array([[cos, 0, -1 * sin, 0],
                                 [0, 1, 0, 0],
                                 [sin, 0, cos, 0],
                                 [0, 0, 0, 1]])
        if direction == 'z':
            self.angle += theta
            rotation = np.array([[cos, -1 * sin, 0, 0],
                                 [sin, cos, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        self.matrix = self.matrixMult(rotation)

    def translate(self, x, y, z):
        self.matrix[0, 3] += x
        self.matrix[1, 3] += y
        self.matrix[2, 3] += z

    def matrixMult(self, matrix):
        result = np.zeros([self.dimensions, self.dimensions])
        for x in range(self.dimensions):  # dot product
            for y in range(self.dimensions):
                product = 0
                for m in range(self.dimensions):
                    product += self.matrix[x, m] * matrix[m, y]
                result[x, y] = product
        return (result)

    def inverse(self):
        inverse = self.minorsMatrix(self.matrix)
        inverse = self.cofactors(inverse)
        inverse = self.adjucate(inverse)
        inverse = self.multiplyConst(inverse, (1 / self.determinant(self.matrix)))
        return (inverse)

    def multiplyConst(self, matrix, const):
        size = int(np.sqrt(matrix.size))
        product = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                product[x, y] = const * matrix[x, y]
        return (product)

    def adjucate(self, matrix):
        size = int(np.sqrt(matrix.size))
        adjucate = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                adjucate[x, y] = matrix[y, x]
        return (adjucate)

    def cofactors(self, matrix):
        size = int(np.sqrt(matrix.size))
        cofactors = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                if x % 2 == 1:
                    cofactors[x, y] = -1 * matrix[x, y]
                else:
                    cofactors[x, y] = matrix[x, y]
                if y % 2 == 1:
                    cofactors[x, y] = -1 * cofactors[x, y]
        return (cofactors)

    def minorsMatrix(self, matrix):
        size = int(np.sqrt(matrix.size))
        minors = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                submatrix = np.zeros((size - 1, size - 1))
                m = 0
                for x1 in range(size):
                    if x1 != x:
                        n = 0
                        for y1 in range(size):
                            if y1 != y:
                                submatrix[m, n] = matrix[x1, y1]
                                n += 1
                        m += 1
                minors[x, y] = self.determinant(submatrix)
        return (minors)

    def determinant(self, matrix):
        determinant = 0
        size = int(np.sqrt(matrix.size))
        if size > 2:
            for x in range(size):
                submatrix = np.zeros((size - 1, size - 1))
                n = 0
                for y in range(size):
                    if y != x:
                        for m in range(1, size):
                            submatrix[m - 1, n] = matrix[m, y]
                        n += 1
                if (x % 2) == 0:
                    determinant += matrix[0, x] * self.determinant(submatrix)
                else:
                    determinant -= matrix[0, x] * self.determinant(submatrix)
        else:
            determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        return (determinant)


# Point and vector classes
class point:

    def __init__(self, x, y, z):
        self.point = np.array([[x],
                               [y],
                               [z],
                               [1]])

    def addVector(self, vector):
        pointFinal = np.array([0, 0, 0])
        pointFinal[0] = self.point[0] + vector[0]
        pointFinal[1] = self.point[1] + vector[1]
        pointFinal[2] = self.point[2] + vector[2]
        return (pointFinal)

    def subtractVector(self, vector):
        pointFinal = np.array([0, 0, 0])
        pointFinal[0] = self.point[0] - vector[0]
        pointFinal[1] = self.point[1] - vector[1]
        pointFinal[2] = self.point[2] - vector[2]
        return (pointFinal)

    def subtractPoint(self, point):
        vectorFinal = np.array([0, 0, 0])
        vectorFinal[0] = self.point[0] - point[0]
        vectorFinal[1] = self.point[1] - point[1]
        vectorFinal[2] = self.point[2] - point[2]
        return (vectorFinal)

    def transform(self, matrix):
        dimensions = int(np.sqrt(matrix.size))
        result = np.zeros((dimensions, 1))
        for x in range(dimensions):
            for y in range(dimensions):
                result[x, 0] += matrix[x, y] * self.point[y, 0]
        self.point = result

    def draw(self, screen):
        z = self.point[2, 0]
        if z <= -1:
            x = self.point[0, 0] / z
            y = self.point[1, 0] / z
            if x < canvasx / 2 and x > -(canvasx / 2) and y < 0.5 and y > -0.5:
                x = int(x / (canvasx / 2) * (WIDTH / 2) + (WIDTH / 2))
                y = int((HEIGHT / 2) - (2 * y) * (HEIGHT / 2))
                pygame.draw.circle(screen, black, (x, y), 6)


class vector:

    def __init__(self, x, y, z):
        self.vector = np.array([x, y, z])

    def addVector(self, vector):
        vectorFinal = np.array([0, 0, 0])
        vectorFinal[0] = self.vector[0] + vector[0]
        vectorFinal[1] = self.vector[1] + vector[1]
        vectorFinal[2] = self.vector[2] + vector[2]
        return (vectorFinal)

    def subtractVector(self, vector):
        vectorFinal = np.array([0, 0, 0])
        vectorFinal[0] = self.vector[0] - vector[0]
        vectorFinal[1] = self.vector[1] - vector[1]
        vectorFinal[2] = self.vector[2] - vector[2]
        return (vectorFinal)


# Initializes pygame
def initScreen():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Test Window")
    screen.fill(white)
    return (screen)


def run():
    done = False
    numPoints = points.shape[0]
    translation = Transformation(4)
    rotationx = Transformation(4)
    rotationx.angle = 0
    rotationy = Transformation(4)
    rotationy.angle = 90 * np.pi / 180
    rotationz = Transformation(4)
    rotationz.angle = 180 * np.pi / 180
    rotationz.rotate('z', 180 * np.pi / 180)
    translation.translate(0, 0, 5)
    screen = initScreen()
    while not done:
        # Main event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        fps = time.time()
        screen.fill(white)
        loop(numPoints, screen, translation, rotationx, rotationy, rotationz)
        pygame.display.flip()
        fps = time.time() - fps
        fps = 1 / fps
        print("fps:" + str(fps))


def loop(numPoints, screen, translation, rotationx, rotationy, rotationz):
    # Keyboard inputs modify separate matrices for each axis
    timer = time.time()
    if keyboard.is_pressed('w'):
        rotationx.rotate('x', 2 * np.pi / 180)
    if keyboard.is_pressed('s'):
        rotationx.rotate('x', -2 * np.pi / 180)
    if keyboard.is_pressed('a'):
        rotationy.rotate('y', -2 * np.pi / 180)
    if keyboard.is_pressed('d'):
        rotationy.rotate('y', 2 * np.pi / 180)
    if keyboard.is_pressed('down arrow'):
        y = np.sin(rotationx.angle) * maxSpeed
        x = np.cos(rotationy.angle) * np.cos(rotationx.angle) * maxSpeed
        z = np.sin(rotationy.angle) * np.cos(rotationx.angle) * maxSpeed
        translation.translate(x, y, z)
    if keyboard.is_pressed('up arrow'):
        y = -1 * (np.sin(rotationx.angle) * maxSpeed)
        x = -1 * (np.cos(rotationy.angle) * np.cos(rotationx.angle) * maxSpeed)
        z = -1 * (np.sin(rotationy.angle) * np.cos(rotationx.angle) * maxSpeed)
        translation.translate(x, y, z)
    if keyboard.is_pressed('right arrow'):
        y = 0
        x = np.cos(rotationy.angle + (np.pi / 2)) * maxSpeed
        z = np.sin(rotationy.angle + (np.pi / 2)) * maxSpeed
        translation.translate(x, y, z)
    if keyboard.is_pressed('left arrow'):
        y = 0
        x = np.cos(rotationy.angle - (np.pi / 2)) * maxSpeed
        z = np.sin(rotationy.angle - (np.pi / 2)) * maxSpeed
        translation.translate(x, y, z)
    print("input:" + str(time.time() - timer))

    # Matrices are combined
    timer = time.time()
    camera = Transformation(4)
    camera.matrix = camera.matrixMult(translation.matrix)
    camera.matrix = camera.matrixMult(rotationy.matrix)
    camera.matrix = camera.matrixMult(rotationx.matrix)
    camera.matrix = camera.matrixMult(rotationz.matrix)

    print("combine:" + str(time.time() - timer))

    timer = time.time()
    inverse = camera.inverse()
    print("invert:" + str(time.time() - timer))

    timer = time.time()
    for x in range(numPoints):
        Point = point(points[x, 0], points[x, 1], points[x, 2])
        Point.transform(inverse)
        Point.draw(screen)
    print("draw:" + str(time.time() - timer))


run()