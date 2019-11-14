#!/usr/bin/env python
#coding: utf-8
from evdev import InputDevice
from select import select
import threading



class keyboard(threading.Thread):
    foward = 103
    left = 105
    right = 106
    back = 108
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("Starting " + self.name)
        self.detectInputKey()
        print("Exiting " + self.name)

    def detectInputKey(self):
        dev = InputDevice('/dev/input/event4')
        while True:
            select([dev], [], [])
            for event in dev.read():
                if (event.value == 1) and event.code != 0:
                    # print("code:%s value:%s" % (event.code, event.value))
                    if event.code is self.left:
                        print("left")
                    elif event.code is self.right:
                        print("right")
                    elif event.code is self.foward:
                        print("forward")
                    elif event.code is self.back:
                        print("back")





if __name__ == '__main__':
    thread1 = keyboard(1, "Thread-1", 1)
    thread1.start()