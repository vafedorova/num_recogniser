from tkinter import Tk, Canvas, Label, Button
import numpy as np
import neuronet
import pickle
import csv
import random

ROWS = 28
COLS = 28
SZ = 10

image = np.zeros((ROWS, COLS))
color_codes = {"black":0, "azure1":255, "azure2":238, "azure3":205, "azure4":139}
def put_pixel(x, y, col):
    global image
    if (image[y // SZ][x // SZ] != 1):
        canvas.create_rectangle(x // SZ * SZ, y // SZ * SZ, x // SZ * SZ + SZ, y // SZ * SZ + SZ, fill=col)
        image[y // SZ][x // SZ] = (255 - color_codes[col]) / 255

def draw(event):
    put_pixel(event.x, event.y, "black")
    put_pixel(event.x , event.y - SZ, "azure" + str(random.randint(2, 4)))
    put_pixel(event.x + SZ, event.y, "azure" + str(random.randint(2, 4)))
    put_pixel(event.x, event.y + SZ, "azure" + str(random.randint(2, 4)))
    put_pixel(event.x - SZ, event.y, "azure" + str(random.randint(2, 4)))

def predict(event):
    result["text"] = np.argmax(net.prediction(image.flatten()))

def clear_image(event):
    global image
    with open('image.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in image:
            writer.writerow(row)
    canvas.delete("all")
    image = np.zeros((ROWS, COLS))
    result["text"] = "?"

root = Tk()

canvas = Canvas(root, width=SZ*COLS, height=SZ*ROWS)
canvas.grid(row=0, column=0, rowspan=3)

go = Button(text="Predict")
go.grid(row=0, column=1)

result = Label(root, text="?", font=("Courier", 100))
result.grid(row=1, column=1)

clear = Button(text="Clear")
clear.grid(row=2, column=1)

canvas.bind("<B1-Motion>", draw)
go.bind("<Button-1>", predict)
clear.bind("<Button-1>", clear_image)

input_nodes = ROWS * COLS
hidden_nodes = 270
output_nodes = 10
w0 = []
w1 = []
with open('w0.pickle','rb') as f:
    w0 = pickle.load(f)
with open('w1.pickle','rb') as f:
    w1 = pickle.load(f)
net = neuronet.neuronet(input_nodes, hidden_nodes, output_nodes, w0, w1)

root.mainloop()