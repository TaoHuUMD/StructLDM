from termcolor import colored

def printy(*arg):
    print(colored(arg,'yellow'))

def prints(a, s=""):
    print("************")
    if a is None: 
        print("None " + s)
        return
    print(a.shape, s)
    print("************")

def printd(*arg):
    print("************")
    printy(arg)
    print("************")

def printb(*arg):
    print(colored(arg,'blue'))

def printg(*arg):
    print(colored(arg,'green'))

def printr(*arg):
    print(colored(arg,'red'))


def print_data(data, s=""):

    if isinstance(data, tuple):
        printg(f"{s} tuple ", len(data)) 
        for d in data: print_data(d)
    elif isinstance(data, list):
        printg(f"{s} list ", len(data)) 
        for d in data: print_data(d)
    elif isinstance(data, dict):
        printg(f"{s} dict ", data.keys()) 
        for d in data.keys(): print_data(data[d])
    elif hasattr(data, '__len__') and (not isinstance(data, str)): #ndarray
        printy(f"{s} arr ", type(data), data.shape)
    else:
        printy(f"{s} scalar ", data)

