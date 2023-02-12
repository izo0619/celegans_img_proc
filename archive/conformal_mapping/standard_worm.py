import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat

def gen():
    # go counterclockwise !!!!!!
    x_st = 125
    y_st = 625
    num_piece = 10
    radius = 25
    length = 25*2*20/num_piece
    num_pts = 50
    pieces = []
    # head 
    theta = np.linspace(np.pi/2, 3*np.pi/2, num_pts)
    x_head = radius * np.cos(theta) + (radius + x_st)
    y_head = radius * np.sin(theta) + y_st
    # bottom side
    x_head = np.concatenate((x_head, np.linspace(x_st + length, radius + x_st, num_pts)), axis=0) 
    y_head = np.concatenate((y_head, np.full(num_pts, y_st - radius)), axis=0) 
    # right side
    x_head = np.concatenate((x_head, np.full(num_pts, x_st + length)), axis=0) 
    y_head = np.concatenate((y_head, np.linspace(y_st - radius, y_st + radius, num_pts)), axis=0)
    # upper edge
    x_head = np.concatenate((x_head, np.linspace(radius + x_st, x_st + length, num_pts)), axis=0) 
    y_head = np.concatenate((y_head, np.full(num_pts, y_st + radius)), axis=0) 
    head = np.empty(np.shape(x_head), dtype=np.complex128); head.real = x_head; head.imag = y_head
    pieces.append(head)

    for i in range(1,num_piece-1):
        # left side 
        x_1 = np.full(num_pts, x_st + i*length)
        y_1 = np.linspace(y_st - radius, y_st + radius, num_pts)
        # bottom side
        x_1 = np.concatenate((x_1, np.linspace(x_st + i*length, x_st + (i+1)*length, num_pts)), axis=0) 
        y_1 = np.concatenate((y_1, np.full(num_pts, y_st - radius)), axis=0) 
        # right side
        x_1 = np.concatenate((x_1, np.full(num_pts, x_st + (i+1)*length)), axis=0) 
        y_1 = np.concatenate((y_1, np.linspace(y_st - radius, y_st + radius, num_pts)), axis=0)
        # upper edge
        x_1 = np.concatenate((x_1, np.linspace(x_st + (i+1)*length, x_st + i*length, num_pts)), axis=0) 
        y_1 = np.concatenate((y_1, np.full(num_pts, y_st + radius)), axis=0) 
        p1 = np.empty(np.shape(x_1), dtype=np.complex128); p1.real = x_1; p1.imag = y_1
        pieces.append(p1)

    # # left side 
    # x_2 = np.full(num_pts, x_st + 2*length)
    # y_2 = np.linspace(y_st - radius, y_st + radius, num_pts)
    # # bottom side
    # x_2 = np.concatenate((x_2, np.linspace(x_st + 2*length, x_st + 3*length, num_pts)), axis=0) 
    # y_2 = np.concatenate((y_2, np.full(num_pts, y_st - radius)), axis=0) 
    # # right side
    # x_2 = np.concatenate((x_2, np.full(num_pts, x_st + 3*length)), axis=0) 
    # y_2 = np.concatenate((y_2, np.linspace(y_st - radius, y_st + radius, num_pts)), axis=0)
    # # upper edge
    # x_2 = np.concatenate((x_2, np.linspace(x_st + 3*length, x_st + 2*length, num_pts)), axis=0) 
    # y_2 = np.concatenate((y_2, np.full(num_pts, y_st + radius)), axis=0) 

    # # left side 
    # x_3 = np.full(num_pts, x_st + 3*length)
    # y_3 = np.linspace(y_st - radius, y_st + radius, num_pts)
    # # bottom side
    # x_3 = np.concatenate((x_3, np.linspace(x_st + 3*length, x_st + 4*length, num_pts)), axis=0) 
    # y_3 = np.concatenate((y_3, np.full(num_pts, y_st - radius)), axis=0) 
    # # right side
    # x_3 = np.concatenate((x_3, np.full(num_pts, x_st + 4*length)), axis=0) 
    # y_3 = np.concatenate((y_3, np.linspace(y_st - radius, y_st + radius, num_pts)), axis=0)
    # # upper edge
    # x_3 = np.concatenate((x_3, np.linspace(x_st + 4*length, x_st + 3*length, num_pts)), axis=0) 
    # y_3 = np.concatenate((y_3, np.full(num_pts, y_st + radius)), axis=0) 

    # left side 
    x_tail = np.full(num_pts, x_st + (num_piece-1)*length)
    y_tail = np.linspace(y_st - radius, y_st + radius, num_pts)
    # bottom side
    x_tail = np.concatenate((x_tail, np.linspace(x_st + (num_piece-1)*length, x_st + num_piece*length - (num_piece-1)*radius, num_pts)), axis=0) 
    y_tail = np.concatenate((y_tail, np.full(num_pts, y_st - radius)), axis=0) 
    # bottom triangle
    x_tail = np.concatenate((x_tail, np.linspace(x_st + num_piece*length - (num_piece-1)*radius, x_st + num_piece*length, num_pts)), axis=0) 
    y_tail = np.concatenate((y_tail, np.linspace(y_st - radius, y_st, num_pts)), axis=0) 
    # top triangle
    x_tail = np.concatenate((x_tail, np.linspace(x_st + num_piece*length, x_st + num_piece*length - (num_piece-1)*radius, num_pts)), axis=0) 
    y_tail = np.concatenate((y_tail, np.linspace(y_st, y_st + radius, num_pts)), axis=0) 
    # upper edge
    x_tail = np.concatenate((x_tail, np.linspace(x_st + num_piece*length - (num_piece-1)*radius, x_st + (num_piece-1)*length, num_pts)), axis=0) 
    y_tail = np.concatenate((y_tail, np.full(num_pts, y_st + radius)), axis=0) 

    # # for small pieces
    # # left side 
    # x_tail = np.full(num_pts, x_st + (num_piece-1)*length)
    # y_tail = np.linspace(y_st - radius, y_st + radius, num_pts)
    # # bottom triangle
    # x_tail = np.concatenate((x_tail, np.linspace(x_st + (num_piece-1)*length, x_st + num_piece*length, num_pts)), axis=0) 
    # y_tail = np.concatenate((y_tail, np.linspace(y_st - radius, y_st, num_pts)), axis=0) 
    # # top triangle
    # x_tail = np.concatenate((x_tail, np.linspace(x_st + num_piece*length, x_st + (num_piece-1)*length, num_pts)), axis=0) 
    # y_tail = np.concatenate((y_tail, np.linspace(y_st, y_st + radius, num_pts)), axis=0) 

    
    # p1 = np.empty(np.shape(x_1), dtype=np.complex128); p1.real = x_1; p1.imag = y_1
    # p2 = np.empty(np.shape(x_2), dtype=np.complex128); p2.real = x_2; p2.imag = y_2
    # p3 = np.empty(np.shape(x_3), dtype=np.complex128); p3.real = x_3; p3.imag = y_3
    tail = np.empty(np.shape(x_tail), dtype=np.complex128); tail.real = x_tail; tail.imag = y_tail
    pieces.append(tail)

    plt.plot(x_head, y_head)
    for piece in pieces:
        plt.plot(piece.real, piece.imag)
    plt.plot(x_tail, y_tail)
    plt.axis('equal')
    plt.title('Standard worm')
    plt.show()
    # savemat('standardworm6' + '.mat', {'cpiece': np.array(pieces,dtype=object)})
    return pieces

gen()