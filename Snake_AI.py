import time, copy, multiprocessing, threading, numpy as np, Network as net, Snake as s, random
from tkinter import *
from tkinter import ttk

brain = [7, 20, 3]
amount = 1000
mutation_amount = 0.1
mutation_chance = 0.01
chosen = 0.25
game_size = 15
creatures = [net.network(brain) for i in range(amount)]

class game_info():
    def __init__(self):
        self.creature = copy.deepcopy(creatures[0])
        self.game = s.snake(game_size)
        self.done = False
        self.direction = 1
        self.counter = 0

def bubblesort(lis):
    move = True
    while(move):
        move = False
        for i in range(len(lis)-1):
            if(lis[i][0] < lis[i+1][0]):
                temp = lis[i][0]
                lis[i][0] = lis[i+1][0]
                lis[i+1][0] = temp

                temp = lis[i][1]
                lis[i][1] = lis[i+1][1]
                lis[i+1][1] = temp

                move = True
    return lis

def create_data(game, direction):
    out_data = [0]*brain[0]

    #Checking distance in all directions
    dist_y_top = 0
    f_y_top = 0
    for i in range(game.snake[0][1]-1, -1, -1):
        if(game.block[game.snake[0][0]][i] == "O"):
            break
        if(game.block[game.snake[0][0]][i] == "X"):
            f_y_top = 1
            break
        dist_y_top += 1

    dist_y_bot = 0
    f_y_bot = 0
    for i in range(game.snake[0][1]+1, game.size):
        if(game.block[game.snake[0][0]][i] == "O"):
            break
        if(game.block[game.snake[0][0]][i] == "X"):
            f_y_bot = 1
            break
        dist_y_bot += 1

    dist_x_right = 0
    f_x_right = 0
    for i in range(game.snake[0][0]+1, game.size):
        if(game.block[i][game.snake[0][1]] == "O"):
            break
        if(game.block[i][game.snake[0][1]] == "X"):
            f_x_right = 1
            break
        dist_x_right += 1

    dist_x_left = 0
    f_x_left = 0
    for i in range(game.snake[0][0]-1, -1, -1):
        if(game.block[i][game.snake[0][1]] == "O"):
            break
        if(game.block[i][game.snake[0][1]] == "X"):
            f_x_left = 1
            break
        dist_x_left += 1

    dist_y_top /= (game.size)
    dist_y_bot /= (game.size)
    dist_x_left /= (game.size)
    dist_x_right /= (game.size)

    dist_y_top = 1 - dist_y_top
    dist_y_bot = 1 - dist_y_bot
    dist_x_left = 1 - dist_x_left
    dist_x_right = 1 - dist_x_right

    # Setting the direction the snake can see
    if(direction == 0):
        out_data[0] = dist_x_left
        out_data[1] = dist_y_top
        out_data[2] = dist_x_right

        if(f_x_left == 1):
            out_data[0] = -1
        elif(f_y_top == 1):
            out_data[1] = -1
        elif(f_x_right == 1):
            out_data[2] = -1

    elif(direction == 1):
        out_data[0] = dist_y_top
        out_data[1] = dist_x_right
        out_data[2] = dist_y_bot

        if(f_y_top == 1):
            out_data[0] = -1
        elif(f_x_right == 1):
            out_data[1] = -1
        elif(f_y_bot == 1):
            out_data[2] = -1

    elif(direction == 2):
        out_data[0] = dist_x_right
        out_data[1] = dist_y_bot
        out_data[2] = dist_x_left

        if(f_x_right == 1):
            out_data[0] = -1
        elif(f_y_bot == 1):
            out_data[1] = -1
        elif(f_x_left == 1):
            out_data[2] = -1

    elif(direction == 3):
        out_data[0] = dist_y_bot
        out_data[1] = dist_x_left
        out_data[2] = dist_y_top

        if(f_y_bot == 1):
            out_data[0] = -1
        elif(f_x_left == 1):
            out_data[1] = -1
        elif(f_y_top == 1):
            out_data[2] = -1

    #Relative location from the end to the fruit #Can be seen as smell/sense
    #X-axis
    out_data[3] = (game.snake[0][0] - game.fruit[0])/(game.size-1)
    #Y-axis
    out_data[4] = (game.snake[0][1] - game.fruit[1])/(game.size-1)

    #Relative location from the end to the foot of the snake
    #X-axis
    out_data[5] = (game.snake[0][0] - game.snake[len(game.snake)-1][0])/(game.size-1)
    #Y-axis
    out_data[6] = (game.snake[0][1] - game.snake[len(game.snake)-1][1])/(game.size-1)

    return out_data

def play_game(input_creatures, poz):
    score_list = [None for i in range(len(input_creatures))]
    #playing all the creatures
    for creature in range(len(input_creatures)):
        data = [None]*3
        score_total = 0

        #Playing 5 games per network
        for time_played in range(5):
            score = 0
            game = s.snake(game_size)
            direction = 1

            for move in range(100):

                #Snake thinking
                snake_input = create_data(game, direction)
                data = input_creatures[creature].forward(snake_input, "relu")

                new_data = [[data[i], i] for i in range(3)]
                new_data = bubblesort(new_data)

                #turn left
                if(new_data[0][1] == 0):
                    if(the_move != 0):
                        the_move = 0
                    direction -= 1
                    if(direction < 0):
                        direction = 3

                #turn right
                elif(new_data[0][1] == 2):
                    if(the_move != 2):
                        the_move = 2
                    direction += 1
                    if(direction > 3):
                        direction = 0

                if(game.move(direction)):
                    break
                score = len(game.snake)

            score_total += (1-((game.snake[0][0] - game.fruit[0])**2 + (game.snake[0][1] - game.fruit[1])**2)/(game.size))/5
            score_total += score

        score_list[creature] = [score_total, creature+poz*len(input_creatures)]
    return score_list

def play_game_async(creature):
    data = [None]*3
    score_total = 0

    #Playing 5 games per network
    for time_played in range(3):
        score = 0
        game = s.snake(game_size)
        direction = 1

        #The amount of moves before the snake dies if no fruit is eaten
        for move in range(int(game.size*2)):

            new_data = []
            if(creature.aichoise < random.random()):
                new_data.append([0, random.randrange(0,3)])
            else:
                #Snake thinking
                snake_input = create_data(game, direction)
                data = creature.forward(snake_input, "sigmoid")

                new_data = [[data[i], i] for i in range(3)]
                new_data = bubblesort(new_data)

            #turn left
            if(new_data[0][1] == 0):
                direction -= 1
                if(direction < 0):
                    direction = 3

            #turn right
            elif(new_data[0][1] == 2):
                direction += 1
                if(direction > 3):
                    direction = 0

            if(game.move(direction)):
                break

            if(len(game.snake) > score):
                score = len(game.snake)

        score_total += (1-(((game.snake[0][0] - game.fruit[0])**2 + (game.snake[0][1] - game.fruit[1])**2)**0.5)/(game.size))/5
        score_total += score
    return [score_total, creature.pos]

def evolution_algorithm(gens):
    #Code for the algorithm
    global creatures
    amount_of_workers = 8
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(amount_of_workers)

    for gen in range(gens):
        for i in range(0, len(creatures)):
            creatures[i].pos = i
        print("gen: ", gen)

        #starmap testing this first
        #input_list = [([creatures[j+int(amount/amount_of_workers)*i]for j in range(int(amount/amount_of_workers))], i,) for i in range(int(amount_of_workers))]
        #results = pool.starmap(play_game, input_list)

        #First 100 gens will be the percentage the ai get to make moves
        #moves_ai_percent = 1
        if(gen < 10):
            moves_ai_percent  = gen/10
        for i in range(len(creatures)):
            creatures[i].aichoise = moves_ai_percent

        #async
        results = pool.map_async(play_game_async, creatures, chunksize=40)
        res = bubblesort(results.get())

        sorted_creatures = []
        for i in range(0, len(res)):
            sorted_creatures.append(copy.deepcopy(creatures[res[i][1]]))

        if(gen % 10 == 0):
            print(creatures[0].weight[0][0])
        print(res[0])

        creatures = net.multiply(sorted_creatures, amount, chosen, mutation_chance, mutation_amount, 0.9)

        if(len(creatures) < amount):
            print("Not enough creatures created, stopping evolution")
            exit(0)
    #creatures[0].save("Snake_100_gen")
    pool.close()
    pool.join()

def update(root, box, canvas, info):
    #The code for the graphics
    for y in range(game_size):
        for x in range(game_size):
            if(info.game.block[x][y] == "O"):
                canvas.itemconfigure(box[x][y], fill="Green")
            elif(info.game.block[x][y] == "X"):
                canvas.itemconfigure(box[x][y], fill="Red")
            else:
                canvas.itemconfigure(box[x][y], fill="Gray")

    #It will play a game with the first creature of the generation
    if(info.done == True):
        info.creature = copy.deepcopy(creatures[0])
        info.game = s.snake(game_size)
        info.done = False
        info.counter = 0
        info.direction = 1

    #Play the game for one move
    snake_input = create_data(info.game, info.direction)
    data = info.creature.forward(snake_input, "sigmoid")

    new_data = [[data[i], i] for i in range(3)]
    new_data = bubblesort(new_data)

    #turn left
    if(new_data[0][1] == 0):
        info.direction -= 1
        if(info.direction < 0):
            info.direction = 3

    #turn right
    elif(new_data[0][1] == 2):
        info.direction += 1
        if(info.direction > 3):
            info.direction = 0

    if(info.game.move(info.direction)):
        info.done = True

    if(info.counter > 50):
        info.done = True
    else:
        info.counter+=1

    root.after(100, update, root, box, canvas, info)


def main():

    #Code for the visually side of things
    root = Tk()
    root.geometry(f"860x640")
    canvas = Canvas(root, width="640", height="860")
    canvas.pack()

    #Just a visual size of the boxes
    box = [[None for y in range(game_size)] for x in range(game_size)]
    s = 30
    for y in range(game_size):
        for x in range(game_size):
            box[x][y] = canvas.create_rectangle(s*x, s*y, s*x+s, s*y+s, fill='Gray')

    info = game_info()
    root.after(100, update, root, box, canvas, info)
    root.mainloop()


if __name__ == "__main__":
    evo_process = threading.Thread(target=evolution_algorithm, args=(1000,))
    evo_process.start()

    main()
