import random, time

class snake():
    def __init__(self, size):
        self.size = size
        self.block = [[" " for y in range(size)] for x in range(size)]

        #Snake cordinates first one will be the head
        self.snake = self.create_snake(3)
        self.fruit = self.create_fruit()


    def create_snake(self, lenght):
        arr = [None]*lenght
        r = random.randrange(0,self.size)
        for i in range(lenght):
            arr[i] = [2-i, r]
            self.block[2-i][r] = "O"

        return arr

    #direction 0 = up, 1 = down, 2 = right, 3 = left
    def create_fruit(self):
        x = random.randrange(0, self.size)
        y = random.randrange(0, self.size)
        if(self.block[x][y] == " "):
            self.block[x][y] = "X"
            return [x, y]
        else:
            return self.create_fruit()


    def move(self, direction):

        end_snake = list(self.snake[len(self.snake)-1])

        #CLearing the last block of the snake
        self.block[self.snake[len(self.snake)-1][0]][self.snake[len(self.snake)-1][1]] = " "

        #Moving all the cordinates to the one in fron of the snake exept for the head
        for i in range(len(self.snake)-1, 0,-1):
            self.snake[i] = list(self.snake[i-1])

        #Moving the head
        #Up
        if(direction == 0 and self.snake[0][1] > 0):
            self.snake[0][1] -= 1
        #Down
        elif(direction == 2 and self.snake[0][1] < self.size-1):
            self.snake[0][1] += 1
        #Right
        elif(direction == 1 and self.snake[0][0] < self.size-1):
            self.snake[0][0] += 1
        #Left
        elif(direction == 3 and self.snake[0][0] > 0):
            self.snake[0][0] -= 1
        else:
            return True

        #If the fruit is eaten
        if(self.snake[0] == self.fruit):
            self.snake.append(end_snake)
            self.block[end_snake[0]][end_snake[1]] = "O"
            self.fruit = self.create_fruit()

        if(self.block[self.snake[0][0]][self.snake[0][1]] == "O"):
            return True

        #Drawing the new head
        self.block[self.snake[0][0]][self.snake[0][1]] = "O"
        return False
