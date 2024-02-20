import os
import tkinter as tk

wind = tk.Tk()

l1 = tk.Label(wind, text = 'Model')
l2 = tk.Label(wind, text = 'Batch size')
l3 = tk.Label(wind, text = 'Learning rate')
l4 = tk.Label(wind, text = 'Number of epochs')
l5 = tk.Label(wind, text = 'Seed')
l6 = tk.Label(wind, text = 'Optimizer')


e1 = tk.Entry(wind)                 # should enter a string here
e2 = tk.Entry(wind)
e3 = tk.Entry(wind)
e4 = tk.Entry(wind)
e5 = tk.Entry(wind)
e6 = tk.Entry(wind)                 # should enter an optimizer name from optax here as a string

l1.grid(row=0, column=0)
l2.grid(row=1, column=0)
l3.grid(row=2, column=0)
l4.grid(row=3, column=0)
l5.grid(row=4, column=0)
l6.grid(row=5, column=0)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)

cwd = os.getcwd()
path_param = os.path.join(cwd,'parameters')
path_param_module = os.path.join(path_param,'gui_parameters_temp.py')


def SaveCallback():    

    with open(path_param_module, 'w') as out:                            # gui_parameters_temp.py is the module with the values of the parameters as entered in the gui 
                                                                         # rewritten every time new parameters are entered, mnist_main.py reads parameter values from this module
        out.write(f"model={e1.get()}\n")
        out.write(f"batch_size={e2.get()}\n")
        out.write(f"lr={e3.get()}\n")
        out.write(f"epochs={e4.get()}\n")
        out.write(f"seed={e5.get()}\n")
        out.write(f"optimizer={e6.get()}\n")

def close():
    wind.destroy()


cb1 = tk.Button(wind, text='Save', command=SaveCallback)                  # button to press after entering the parameters
cb2 = tk.Button(wind, text='Close', command=close)                        # button to close the window
cb1.grid(row=6, column=0)
cb2.grid(row=6, column=1)

wind.mainloop()

