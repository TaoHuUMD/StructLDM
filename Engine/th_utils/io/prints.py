from termcolor import colored

def printy(*arg):
    print(colored(arg,'yellow'))

def printd(*arg):
    printy(arg)

def printb(*arg):
    print(colored(arg,'blue'))

def printg(*arg):
    print(colored(arg,'green'))

def printr(*arg):
    print(colored(arg,'red'))