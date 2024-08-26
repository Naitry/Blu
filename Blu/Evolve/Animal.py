class Quadtree:
    def __init__(self,
                 boundary: tuple[int, int, int, int],
                 capacity: int):
        self.boundary: tuple[int] = boundary
        self.capacity = capacity
        self.points: list[tuple[int]] = []
        self.divided: bool = False

    def subdivide(self):
        x: int
        y: int
        w: int
        h: int
        x, y, w, h = self.boundary
        nw: int
        ne: int
        sw: int
        se: int
        nw = (x, y, w / 2, h / 2)
        ne = (x + w / 2, y, w / 2, h / 2)
        sw = (x, y + h / 2, w / 2, h / 2)
        se = (x + w / 2, y + h / 2, w / 2, h / 2)

        self.northwest: Quadtree = Quadtree(nw, self.capacity)
        self.northeast: Quadtree = Quadtree(ne, self.capacity)
        self.southwest: Quadtree = Quadtree(sw, self.capacity)
        self.southeast: Quadtree = Quadtree(se, self.capacity)

        self.divided = True

    def insert(self,
               point: tuple[int, int]):
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

    def contains(self,
                 point: tuple[int]):
        x: int
        y: int
        w: int
        h: int
        x, y, w, h = self.boundary
        px: int = point[0]
        py: int = point[1]
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

    def intersects(self,
                   rangeRect: tuple[int, int, int, int]):
        x: int
        y: int
        w: int
        h: int
        x, y, w, h = self.boundary
        rx, ry, rw, rh = rangeRect
        return not (rx > x + w or rx + rw < x or ry > y + h or ry + rh < y)

    def pointInRect(self,
                    point: tuple[int, int],
                    rect: tuple[int, int, int, int]) -> bool:
        px: int = point[0]
        py: int = point[1]
        rx: int
        ry: int
        rw: int
        rh: int
        rx, ry, rw, rh = rect
        return rx <= px < rx + rw and ry <= py < ry + rh


def drawAnimal(animal: Animal) -> None:
    x, y = animal.position
    GL.glColor3f(1.0, 0.0, 0.0)  # Red color for animals
    GL.glRectf(x, y, x, y)
