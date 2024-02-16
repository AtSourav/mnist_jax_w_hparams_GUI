import os
import tkinter as tk

wind = tk.Tk()

l1 = tk.Label(wind, text = 'Model')
l2 = tk.Label(wind, text = 'Batch size')
l3 = tk.Label(wind, text = 'Learning rate')
l4 = tk.Label(wind, text = 'Number of epochs')
l5 = tk.Label(wind, text = 'Key')


e1 = tk.Entry(wind)
e2 = tk.Entry(wind)
e3 = tk.Entry(wind)
e4 = tk.Entry(wind)
e5 = tk.Entry(wind)

l1.grid(row=0, column=0)
l2.grid(row=1, column=0)
l3.grid(row=2, column=0)
l4.grid(row=3, column=0)
l5.grid(row=4, column=0)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)

def DoneCallback():
    os.makedirs('/parameters/', exist_ok=True)                           # if the folder already exists, then it does nothing

    with open("/parameters/gui_parameters_temp.py", 'w') as out:         # gui_parameters_temp.py acts as the module with the values of the parameters as entered in the gui 
                                                                         # rewritten every time new parameters are entered, mnist_main.py reads parameter values from this module
        out.write(f"model={e1.get()}\n")
        out.write(f"batch_size={e2.get()}\n")
        out.write(f"lr={e3.get()}\n")
        out.write(f"epochs={e4.get()}\n")
        out.write(f"key={e5.get()}\n")


cb = tk.Button(wind, text='Done', command=DoneCallback)                  # button to press Done after entering the parameters
cb.grid(row=5, column=1)

wind.mainloop()

