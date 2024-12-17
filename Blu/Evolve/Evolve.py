from Animal import Animal, ExampleAnimal
import torch

import OpenGL.GL as GL
import OpenGL.GLUT as GLUT


class Message:
    def __init__(self,
                 sender: 'Animal',
                 content: torch.Tensor):
        self.sender = sender
        self.content = content


# Common environment class
class Environment:
    def __init__(self, dimensions: tuple[int, int]):
        self.dimensions = dimensions
        self.animals = []

    def addAnimal(self, animal: Animal) -> None:
        self.animals.append(animal)

    def update(self) -> None:
        for animal in self.animals:
            animal.move(self)
            animal.interact(self)
            # Example learning step: using random input and target data
            input_data = torch.randn(10)
            target_data = torch.randn(2)  # Target is movement (Δx, Δy)
            animal.learn(input_data, target_data)



def drawAnimal(animal: Animal) -> None:
    x, y = animal.position
    GL.glColor3f(1.0, 0.0, 0.0)  # Red color for animals
    GL.glRectf(x, y, x, y)


def display() -> None:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    for animal in environment.animals:
        drawAnimal(animal)
    GLUT.glutSwapBuffers()


def update(value: int) -> None:
    environment.update()
    GLUT.glutPostRedisplay()
    GLUT.glutTimerFunc(100, update, 0)

# OpenGL setup


def initOpenGL(dimensions: tuple[int, int]) -> None:
    GLUT.glutInit()
    GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_RGB)
    GLUT.glutInitWindowSize(dimensions[0] * 10, dimensions[1] * 10)  # Scale up the window size for better visibility
    GLUT.glutCreateWindow("Animal Simulation")
    GLUT.glClearColor(1.0, 1.0, 1.0, 1.0)
    GLUT.glMatrixMode(GL.GL_PROJECTION)
    GLUT.glLoadIdentity()
    GLUT.gluOrtho2D(0, dimensions[0], 0, dimensions[1])
    GLUT.glutDisplayFunc(display)
    GLUT.glutTimerFunc(100, update, 0)


if __name__ == "__main__":
    environment = Environment((1000, 1000))
    example_animal1 = ExampleAnimal(energy=100.0, position=(50, 50), costWeight=0.01, learningRate=0.01)
    example_animal2 = ExampleAnimal(energy=100.0, position=(20, 20), costWeight=0.01, learningRate=0.01)
    environment.addAnimal(example_animal1)
    environment.addAnimal(example_animal2)

    initOpenGL(environment.dimensions)
    GLUT.glutMainLoop()
