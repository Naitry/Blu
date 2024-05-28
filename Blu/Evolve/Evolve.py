import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class Quadtree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary  # Boundary is a rectangle (x, y, width, height)
        self.capacity = capacity  # Capacity is the maximum number of points per node
        self.points = []
        self.divided = False

    def subdivide(self):
        x, y, w, h = self.boundary
        nw = (x, y, w / 2, h / 2)
        ne = (x + w / 2, y, w / 2, h / 2)
        sw = (x, y + h / 2, w / 2, h / 2)
        se = (x + w / 2, y + h / 2, w / 2, h / 2)
        self.northwest = Quadtree(nw, self.capacity)
        self.northeast = Quadtree(ne, self.capacity)
        self.southwest = Quadtree(sw, self.capacity)
        self.southeast = Quadtree(se, self.capacity)
        self.divided = True

    def insert(self, point):
        if not self.contains(point):
            return False

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        else:
            if not self.divided:
                self.subdivide()

            if self.northwest.insert(point):
                return True
            elif self.northeast.insert(point):
                return True
            elif self.southwest.insert(point):
                return True
            elif self.southeast.insert(point):
                return True

        return False

    def contains(self, point):
        x, y, w, h = self.boundary
        px, py = point[0], point[1]
        return x <= px < x + w and y <= py < y + h

    def query(self, range_rect, found):
        if not self.intersects(range_rect):
            return

        for point in self.points:
            if self.point_in_rect(point, range_rect):
                found.append(point)

        if self.divided:
            self.northwest.query(range_rect, found)
            self.northeast.query(range_rect, found)
            self.southwest.query(range_rect, found)
            self.southeast.query(range_rect, found)

    def intersects(self, range_rect):
        x, y, w, h = self.boundary
        rx, ry, rw, rh = range_rect
        return not (rx > x + w or rx + rw < x or ry > y + h or ry + rh < y)

    def point_in_rect(self, point, rect):
        px, py = point[0], point[1]
        rx, ry, rw, rh = rect
        return rx <= px < rx + rw and ry <= py < ry + rh


# Message class to represent communication between animals
class Message:
    def __init__(self, sender: 'Animal', content: torch.Tensor):
        self.sender = sender
        self.content = content

# Base class for animals
class Animal(nn.Module):
    def __init__(self, energy: float, position: tuple[int, int], costWeight: float, learningRate: float = 0.01):
        super(Animal, self).__init__()
        self.energy = energy
        self.position = position
        self.costWeight = costWeight
        self.messages = []
        self.optimizer = None  # Will be set in subclasses
        self.lossFunction = nn.MSELoss()  # Example loss function, can be customized

    def forward(self, input_data: torch.Tensor, messages: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("This method should be overridden by subclasses")

    def interact(self, environment: 'Environment') -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def move(self, environment: 'Environment') -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def consumeEnergy(self, amount: float) -> None:
        self.energy -= amount
        if self.energy <= 0:
            print(f"{self.__class__.__name__} has no energy left!")

    def calculateCost(self, input_data: torch.Tensor) -> float:
        # Example cost calculation based on input size and cost weight
        input_size = input_data.numel()  # Number of elements in the input tensor
        return input_size * self.costWeight

    def sendMessage(self, receiver: 'Animal', content: torch.Tensor) -> None:
        message = Message(sender=self, content=content)
        receiver.receiveMessage(message)

    def receiveMessage(self, message: Message) -> None:
        self.messages.append(message)
        print(f"{self.__class__.__name__} received message from {message.sender.__class__.__name__}")

    def learn(self, input_data: torch.Tensor, target: torch.Tensor) -> None:
        if self.optimizer is None:
            return
        # Collect messages
        messages = [message.content for message in self.messages]
        # Clear received messages after processing
        self.messages = []
        # Forward pass
        output = self.forward(input_data, messages)
        # Compute loss
        loss = self.lossFunction(output, target)
        # Backward pass and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f"{self.__class__.__name__} learned with loss: {loss.item():.4f}")

class ExampleAnimal(Animal):
    def __init__(self, energy: float, position: tuple[int, int], costWeight: float, learningRate: float = 0.01):
        super(ExampleAnimal, self).__init__(energy, position, costWeight, learningRate)
        self.fc1 = nn.Linear(20, 50)  # Adjusted input size to include object positions and types
        self.fc2 = nn.Linear(50, 2)  # Output size is 2 for movement (Δx, Δy)
        self.optimizer = optim.SGD(self.parameters(), lr=learningRate)

    def forward(self, nearby_objects: torch.Tensor, input_data: torch.Tensor, messages: list[torch.Tensor]) -> torch.Tensor:
        # Concatenate input data and messages
        messages_tensor = torch.cat(messages) if messages else torch.zeros(10)
        position_tensor = torch.tensor(self.position, dtype=torch.float32)
        combined_input = torch.cat((nearby_objects.flatten(), input_data, position_tensor, messages_tensor))

        x = torch.relu(self.fc1(combined_input))
        output = self.fc2(x)  # Output is the movement (Δx, Δy)

        # Calculate and consume energy cost
        cost = self.calculateCost(combined_input)
        self.consumeEnergy(cost)
        print(f"{self.__class__.__name__} consumed {cost:.2f} energy, remaining energy: {self.energy:.2f}")

        return output

    def interact(self, environment: 'Environment') -> None:
        # Example interaction logic: send a message to another animal
        if environment.animals:
            receiver = random.choice(environment.animals)
            if receiver != self:
                message_content = torch.randn(10)  # Example message content
                self.sendMessage(receiver, message_content)

    def move(self, environment: 'Environment') -> None:
        # Query nearby objects using the quadtree
        range_rect = (self.position[0] - 5, self.position[1] - 5, 10, 10)  # Define the range to query
        nearby_objects = []
        environment.quadtree.query(range_rect, nearby_objects)

        # Convert nearby objects to a tensor (example assumes object is represented by position and type)
        nearby_objects_tensor = torch.tensor([[obj[0], obj[1], obj[2]] for obj in nearby_objects], dtype=torch.float32)

        # Input data for the network (could include various factors, here we use random for simplicity)
        input_data = torch.randn(10)  # Adjusted size
        # Collect messages
        messages = [message.content for message in self.messages]
        # Clear received messages after processing
        self.messages = []
        # Get movement from the network
        movement = self.forward(nearby_objects_tensor, input_data, messages)
        delta_x, delta_y = movement[0].item(), movement[1].item()

        # Update position based on network output
        new_x = self.position[0] + int(delta_x)
        new_y = self.position[1] + int(delta_y)

        # Ensure new position is within environment boundaries
        new_x = max(0, min(new_x, environment.dimensions[0] - 1))
        new_y = max(0, min(new_y, environment.dimensions[1] - 1))

        self.position = (new_x, new_y)
        print(f"{self.__class__.__name__} moved to {self.position}")


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

# OpenGL drawing functions
def drawAnimal(animal: Animal) -> None:
    x, y = animal.position
    glColor3f(1.0, 0.0, 0.0)  # Red color for animals
    glRectf(x, y, x, y)

def display() -> None:
    glClear(GL_COLOR_BUFFER_BIT)
    for animal in environment.animals:
        drawAnimal(animal)
    glutSwapBuffers()

def update(value: int) -> None:
    environment.update()
    glutPostRedisplay()
    glutTimerFunc(100, update, 0)

# OpenGL setup
def initOpenGL(dimensions: tuple[int, int]) -> None:
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(dimensions[0] * 10, dimensions[1] * 10)  # Scale up the window size for better visibility
    glutCreateWindow("Animal Simulation")
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, dimensions[0], 0, dimensions[1])
    glutDisplayFunc(display)
    glutTimerFunc(100, update, 0)

if __name__ == "__main__":
    environment = Environment((1000, 1000))
    example_animal1 = ExampleAnimal(energy=100.0, position=(50, 50), costWeight=0.01, learningRate=0.01)
    example_animal2 = ExampleAnimal(energy=100.0, position=(20, 20), costWeight=0.01, learningRate=0.01)
    environment.addAnimal(example_animal1)
    environment.addAnimal(example_animal2)

    initOpenGL(environment.dimensions)
    glutMainLoop()

