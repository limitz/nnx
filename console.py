import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
import torchvision as _torchvision
import math as _math
import matplotlib.pyplot as _plt
import IPython.display as _ipy_display
import tempfile as _tempfile
import uuid as _uuid
import os as _os
import time as _time
import re as _re


class Timer:
    def __init__(self, style="<green>[<b>{time_str}</b><green>:{seconds}]</>"):
        self.style = style
        self.start = None
        self.value = None

    def restart(self):
        self.start = _time.time()
        
    def __str__(self):
        if self.start is None:
            self.restart()
        self.value = t =_time.time() - self.start
        seconds = int(t % 60)
        minutes = int((t//60) % 60)
        hours = int((t//3600) % 24)
        days = int(t//(24*3600))
        if days > 0:
            time_str = f"{days}d {hours:02d}:{minutes:02d}"
        else:
            time_str = f"{hours:02d}:{minutes:02d}"
        seconds = f"{seconds:02d}"
        return self.style.format(time_str=time_str, seconds=seconds)
        
class ProgressBar:
    def __init__(self, *args, min_value=None, max_value=None, value=None, bar_length=10,
                 title = "",
                 style="<gray>\x00<lime>■<darkgray>■<gray>\x00 {percentage:0.01f}%</>"): # ▒ ▓
        if len(args) == 1:
            assert max_value is None
            self.max_value = args[0]
            self.min_value = min_value or 0
        elif len(args) == 2:
            assert min_value is None and max_value is None
            self.min_value = args[0]
            self.max_value = args[1]
        else:
            self.min_value = min_value or 0
            self.max_value = max_value if max_value is not None else 1
        self.value = value if value is not None else self.min_value
        self.bar_length = bar_length
        style = csi_split(csi_render(style))
        self.style = []
        self.title = title
        working = ""
        for token in style:
            if is_csi_token(token): working += token
            else: 
                self.style.append(working+token)
                working = ""
        assert len(self.style) == 4

    def __str__(self):
        r = ""
        if self.title:
            r += csi_render(self.title) + ": "
        r += self.style[0]
        d = (self.max_value - self.min_value)
        v = (self.value - self.min_value)
        step = d / self.bar_length
        for i in range(self.bar_length):
            if v >= (i * step):
                r += self.style[1]
            else:
                r += self.style[2]
        percentage = (v / d)*100
        r += "".join(self.style[3:])
        kwargs = dict(percentage=percentage, 
                      value=self.value, 
                      min_value=self.min_value,
                      max_value=self.max_value)
        return r.format(**kwargs)
            
class TrueColor:
    def __init__(self):
        self.registered_colors = {}
        self.register("red", "#FF0000")
        self.register("green","#00FF00")
        self.register("blue", "#0000FF")
        self.register("yellow", "#FFFF00")
        self.register("orange", "#FF8800")
        self.register("lime","#88FF00")
        self.register("cyan", "#00FFFF")
        self.register("teal", "#00FF88")
        self.register("magenta","#FF00FF")
        self.register("pink", "#FF4488")
        self.register("purple","#8800FF")
        self.register("white", "#FFFFFF")
        self.register("lightgray", "#BBBBBB")
        self.register("gray", "#777777")
        self.register("darkgray", "#222222")
        self.register("black", "#000000")
        self.registered_colors["/"] = "\x1b[0m"
        self.registered_colors["b"] = "\x1b[1m"
        self.registered_colors["/b"] = "\x1b[22m"
        self.registered_colors["f"] = "\x1b[2m"
        self.registered_colors["/f"] = "\x1b[22m"
        self.registered_colors["i"] = "\x1b[3m"
        self.registered_colors["/i"] = "\x1b[23m"
        self.registered_colors["u"] = "\x1b[4m"
        self.registered_colors["/u"] = "\x1b[24m"
        self.registered_colors["blink"] = "\x1b[6m"
        self.registered_colors["/blink"] = "\x1b[26m"
            
    def register(self, name, rgb):
        c = self.__getitem__(rgb, exclude_prefix=True)
        self.registered_colors[name] = c
        
    def __getitem__(self, color, exclude_prefix=False):
        if isinstance(color, str):
            if "~" in color and not color.startswith("~"):
                split = color.index("~")
                fore, back = color[:split], color[split:]
                fore = self[fore]
                back = self[back]
                return fore+back

            if color.startswith("#"):
                color = color[1:]
                prefix = "\x1b[38"
            elif color.startswith("~"):
                prefix = "\x1b[48"
                color = color[1:]
            else:
                prefix = "\x1b[38"
            
            if exclude_prefix: 
                prefix = ""
            
            if color in self.registered_colors:
                c = self.registered_colors[color]
                if c.startswith("\x1b"): return c
                else: return f"{prefix}{c}"
            
            else:
                if len(color) == 6:
                    r,g,b = (int(color[i*2:2+i*2],base=16) 
                             for i in range(3))
                elif len(color) == 3:
                    r,g,b = (int(color[i*2:2+i*2],base=16) * 17 
                             for i in range(3))
                else:
                    assert False, "expected 3 or 6 hex digits after #" 
                
                return f"{prefix};2;{r};{g};{b}m"
            
        else:
            if isinstance(color, _torch.Tensor):
                color = color.view(-1).unbind(0)
            if isinstance(color, (tuple, list)):
                assert len(color) == 3, "expected sequence of 3 numbers"
                if any(isinstance(v,float) for v in color):
                    color = (int(v*255) for v in color)
                assert all(v>=0 and v<=255 for v in color), "value out of range"
                r,g,b = color
                prefix = "" if exclude_prefix else "\x1b[38"
                return f"{prefix};2;{r};{g};{b}m"
            else:
                assert False, "unsupported type"

color = TrueColor()

def _re_find_color(match):
    name = match.group(1)
    return color[name]

def is_csi_token(c):
    return c.startswith("\x1b") and c.endswith("m")

def csi_split(arg):
    r = []
    while len(arg) > 0:
        if arg[0] == "\x1b":
            end = arg.index("m")
            assert end > 0
            r.append(arg[:end+1])
            arg = arg[end+1:]
        else:
            if "\x1b" in arg:
                end = arg.index("\x1b")
            else:
                end = len(arg)
            r.append(arg[:end])
            arg = arg[end:]
    return r
    
def csi_tokenize(arg):
    r = []
    while len(arg) > 0:
        if arg[0] == "\x1b":
            end = arg.index("m")
            r.append(arg[:end+1])
            arg = arg[end+1:]
        else:
            r.append(arg[0])
            arg = arg[1:]
    return r
    
def csi_render(arg, control="<>"):
    pat = _re.escape(control[0]) + "([\\~/a-zA-Z_0-9\\-]+)" + _re.escape(control[1])
    arg = str(arg)
    arg = _re.sub(pat, _re_find_color, arg)
    return arg
    
_original_print = print
def print(*args, control="<>", **kwargs):
    r = []
    for arg in args:
        r.append(csi_render(arg, control=control))
    r.append(color["/"])
    _original_print(*r, **kwargs)