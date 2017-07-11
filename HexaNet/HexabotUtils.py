import ctypes
import time
import numpy as np
from PIL import Image
import win32gui, win32ui, win32con, win32api
from cv2 import imwrite

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

"""
C struct redefinitions 
"""
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),

                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class MouseInput(ctypes.Structure):

    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

"""
KEY PRESS METHODS
"""
# presses a key
def pressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( hexKeyCode, 0x48, 0, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# releases a key
def releaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( hexKeyCode, 0x48, 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# hits a key for an amount of time
def hitKey(key, holdTime=.02):
    pressKey(key)
    time.sleep(holdTime)
    releaseKey(key)

def fastImageGrab(show=False):
    name = "Super Hexagon"
    window = win32ui.FindWindow(None, name)
    x0,y0,_,_ = window.GetWindowRect()
    hwin = win32gui.GetDesktopWindow()

    width,height,left,top = 768,478,x0+8,y0+30
    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)

    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    im = Image.frombuffer("RGB", (width,height), bmp.GetBitmapBits(True), "raw", "RGBX", 0, 1)
    im = np.array(im)[:,:,:3]
    
    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())
    return im


saveImageTime = int(time.time()*1000)
saveImageFrame = 0
def saveImage():
    global saveImageFrame, saveImageTime
    im = fastImageGrab()
    imwrite("imgs/%d-%d.png"%(saveImageTime,saveImageFrame), im)
    saveImageFrame += 1