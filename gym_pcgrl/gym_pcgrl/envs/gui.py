from pdb import set_trace as TT
import sys

from gym.envs.classic_control.rendering import get_window, get_display
import pyglet
from pyglet import gl
# from pyglet.gui.buttons import Button


# class GUI(object):
#     def __init__(self, display=None, maxwidth=500):
#         self.window = None
#         self.isopen = False
#         self.display = get_display(display)
#         self.maxwidth = maxwidth

#     def imshow(self, arr):
#         if self.window is None:
#             height, width, _channels = arr.shape
#             if width > self.maxwidth:
#                 scale = self.maxwidth / width
#                 width = int(scale * width)
#                 height = int(scale * height)
#             self.window = get_window(
#                 width=width,
#                 height=height,
#                 display=self.display,
#                 vsync=False,
#                 resizable=True,
#             )
#             self.width = width
#             self.height = height
#             self.isopen = True

#             @self.window.event
#             def on_resize(width, height):
#                 self.width = width
#                 self.height = height

#             @self.window.event
#             def on_close():
#                 self.isopen = False

            
#             reset_button = Button()
#             @self.window.event



#         assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
#         image = pyglet.image.ImageData(
#             arr.shape[1], arr.shape[0], "RGB", arr.tobytes(), pitch=arr.shape[1] * -3
#         )
#         texture = image.get_texture()
#         gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
#         texture.width = self.width
#         texture.height = self.height
#         self.window.clear()
#         self.window.switch_to()
#         self.window.dispatch_events()
#         texture.blit(0, 0)  # draw
#         self.window.flip()

#     def close(self):
#         if self.isopen and sys.meta_path:
#             # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
#             self.window.close()
#             self.isopen = False

#     def __del__(self):
#         self.close()


if __name__ == "__main__":
    from tkinter import *
    from tkinter import ttk
    root = Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
    # Add checkbox
    cb = ttk.Checkbutton(frm, text="Checkbox").grid(column=0, row=1)
    for _ in range(100):
        root.mainloop()