from pdb import set_trace as TT
from timeit import default_timer as timer

import gi 
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import numpy as np


#FIXME: sometimes the mere existence of this class will break a multi-env micropolis run
class GtkGUI(Gtk.Window):
    def __init__(self, env, tile_types, tile_images, metrics, metric_trgs, metric_bounds):
        self.env = env
        Gtk.Window.__init__(self, title="Controllable PCGRL")
        self.set_border_width(10)

        self.pixbuf = None
        # This hbox contains the map and the gui side by side
        hbox_0 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        eventbox = Gtk.EventBox()        
        eventbox.connect("motion-notify-event", self.on_mouse_move)
        eventbox.add_events(Gdk.EventMask.POINTER_MOTION_MASK)
        eventbox.connect("button-press-event", self.on_mouse_click)
        eventbox.connect("button-release-event", self.on_mouse_release)
        # self.add(eventbox)        

        image = Gtk.Image()

        eventbox.add(image)
        hbox_0.pack_start(eventbox, False, False, 0)

        # This vbox contains the gui (buttons and control sliders and meters)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        pause_play_box = Gtk.ButtonBox(orientation=Gtk.Orientation.HORIZONTAL)
        # Add pause button
        self.pause_button = Gtk.ToggleButton(label="Pause")
        self.play_button = Gtk.ToggleButton(label="Play")
        self.pause_button.connect("toggled", self.on_pause_toggled)
        self.play_button.connect("toggled", self.on_play_toggled)
        pause_play_box.add(self.pause_button)
        pause_play_box.add(self.play_button)
        vbox.pack_start(pause_play_box, False, False, 0)

        # This hbox contains the auto reset checkbox and the reset button
        hbox_1 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        
        reset_button = Gtk.Button("reset")
        reset_button.connect('clicked', lambda item: self.env.reset())
        hbox_1.pack_start(reset_button, False, False, 0) 

        auto_reset_button = Gtk.CheckButton("auto reset")
        self.env.auto_reset = False
        auto_reset_button.connect('clicked', lambda item: self._enable_auto_reset(item.get_active()))
        hbox_1.pack_start(auto_reset_button, False, False, 0)

        vbox.pack_start(hbox_1, False, False, 0)
        tile_radio_buttons = []
        self._active_tool = tile_types[0]

        hbox_static = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        static_checkbox = Gtk.CheckButton("static builds")
        self._static_build = static_checkbox.get_active()
        static_checkbox.connect('clicked', self.toggle_static_build)
        hbox_static.pack_start(static_checkbox, False, False, 0)
        vbox.pack_start(hbox_static, False, False, 0)


        for tile in tile_types:
            # This hbox contains the tile type name and the tile image
            hbox_t = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            tile_radio_button = Gtk.RadioButton.new_with_label_from_widget(tile_radio_buttons[0] if len(tile_radio_buttons) > 0 else None, tile)
            tile_radio_buttons.append(tile_radio_button)
            tile_radio_button.connect('toggled', self.on_tool_changed, tile)
            if tile_images is not None:
                arr = np.array(tile_images[tile].convert('RGB'))
                tile_image = Gtk.Image()
                shape = arr.shape
                arr = arr.flatten()
                pixbuf = GdkPixbuf.Pixbuf.new_from_data(arr,
                    GdkPixbuf.Colorspace.RGB, False, 8, shape[1], shape[0], 3*shape[1])
                tile_image.set_from_pixbuf(pixbuf)
                hbox_t.pack_start(tile_radio_button, False, False, 0)
                hbox_t.pack_start(tile_image, False, False, 0)
                # hbox_t.pack_start(Gtk.Label(tile), False, False, 0)
                vbox.pack_start(hbox_t, False, False, 0)

        # HACK
        self.trg_bars = {}

        prog_bars = {}
        scales = {}
        prog_labels = {}
        self.metric_ranges = {k: abs(metric_bounds[k][1] - metric_bounds[k][0]) for k in metrics}
        for k in self.env.metrics:
        # if False:
            metric = metrics[k]
            label = Gtk.Label()
            label.set_text(k)
            vbox.pack_start(label, True, True, 0)
            if metric is None:
                metric = 0

            # FIXME: Issues here. Can't find bullet svg, install `conda install -c conda-forge librsvg`, this message 
            #  disappears, then left with segfault.
            # ad = Gtk.Adjustment(metric, metric_bounds[k][0], metric_bounds[k][1],
            #                     # env.param_ranges[k] / 20, env.param_ranges[k] / 10, 0)
            #                     self.metric_ranges[k] / 20, self.metric_ranges[k] / 10, 0)
            # scale = Gtk.HScale(adjustment=ad)
            # scale.set_name(k)
            # scale.set_show_fill_level(True)
            # scales[k] = scale
            # scale.connect("value-changed", self.scale_moved)
            # vbox.pack_start(scale, True, True, 10)

            #HACK: Just use another progress bar for not, smh (no user input!)
            metric_trg = Gtk.ProgressBar()
#           metric_prog.set_draw_value(True)
            self.trg_bars[k] = metric_trg
            vbox.pack_start(metric_trg, True, True, 10)

            prog_label = Gtk.Label()
            prog_label.set_text(str(metric))
            prog_labels[k] = prog_label
            vbox.pack_start(prog_label, True, True, 0)
            metric_prog = Gtk.ProgressBar()
#           metric_prog.set_draw_value(True)
            prog_bars[k] = metric_prog
            vbox.pack_start(metric_prog, True, True, 10)
           #bounds = metric_bounds[k]
           #frac = metrics[k]
           #metric_prog.set_fraction(frac)

        hbox_0.add(vbox) 
        self.add(hbox_0)
       #self.timeout_id = GLib.timeout_add(50, self.on_timeout, None)
       #self.activity_mode = False
        self.image = image
        self.prog_bars = prog_bars
        self.scales = scales
        self.prog_labels = prog_labels
        self._user_clicks = []
        self._tool_down = False
        self._paused = False

    def _enable_auto_reset(self, enable):
        self.env.auto_reset = enable

    def _pause(self):
            self._paused = True
            self.play_button.set_active(False)

    def _play(self):
            self._paused = False
            self.pause_button.set_active(False)

    def on_pause_toggled(self, widget):
        if widget.get_active():
            self._pause()
        else:
            self._play()

    def on_play_toggled(self, widget):
        if widget.get_active():
            self._play()
        else:
            self._pause()

    def on_mouse_move(self, widget, event):
        if self._tool_down:
            self._user_clicks.append((event.x, event.y, self._active_tool, self._static_build))

    def toggle_static_build(self, button):
        self._static_build = button.get_active()

    def on_tool_changed(self, button, tile):
        if not button.get_active():
            return
        self._active_tool = tile

    def on_mouse_click(self, widget, event):
        self._user_clicks.append((event.x, event.y, self._active_tool, self._static_build))
        self._tool_down = True
        # print('click', widget, event.button, event.time)

    def on_mouse_release(self, widget, event):
        self._tool_down = False

    def get_clicks(self):
        user_clicks = self._user_clicks
        self._user_clicks = []
        return user_clicks

    def render(self, img):

        # FIXME: fuxing slow as fuque
        ### PROFILING:
        # N = 100
        # start_time = timer()
        # for _ in range(N):
        #     self.display_image(img)
        # print(f'mean image render time over {N} trials:', (timer() - start_time) * 1000 / N, 'ms')
        # N = 100
        # start_time = timer()
        # for _ in range(N):
        #     self.display_metrics()
        # print(f'mean gui display time over {N} trials:', (timer() - start_time) * 1000 / N, 'ms')
        ###

        self.display_image(img)
        self.display_metrics()
        while Gtk.events_pending():
            Gtk.main_iteration()

    def display_image(self, img):
        # self.image.set_from_file("../Downloads/xbox_cat.png")        
        if img is None:
            return
        shape = img.shape
        arr = img.flatten()
        # self.pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(arr, GdkPixbuf.Colorspace.RGB, False, 8, shape[1], shape[0], shape[2] * shape[1])
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(arr,
            GdkPixbuf.Colorspace.RGB, False, 8, shape[1], shape[0], 3*shape[1])
        # else:
        # if self.pixbuf is None:
        self.image.set_from_pixbuf(pixbuf)
            # self.pixbuf.fill(arr)

    def scale_moved(self, event):
        k = event.get_name()
        self.env.metric_trgs[k] = event.get_value()
        self.env.set_trgs(self.env.metric_trgs)

    def display_metric_trgs(self):
        for k, v in self.env.metric_trgs.items():
            self.trg_bars[k].set_fraction(v / self.metric_ranges[k])

         #TODO:
        # for k, v in self.env.metric_trgs.items():
        #     if k in self.env.ctrl_metrics:
        #         self.scales[k].set_value(v)

    def display_metrics(self):
        for k, prog_bar in self.prog_bars.items():
            metric_val = self.env.metrics[k]
            prog_bar.set_fraction(metric_val / self.metric_ranges[k])
            # prog_bar.set_fraction(metric_val / self.env.param_ranges[k])
            prog_label = self.prog_labels[k]
            prog_label.set_text(str(metric_val))

    def on_show_text_toggled(self, button):
        show_text = button.get_active()
        if show_text:
            text = "some text"
        else:
            text = None
        self.progressbar.set_text(text)
        self.progressbar.set_show_text(show_text)

    def on_activity_mode_toggled(self, button):
        self.activity_mode = button.get_active()
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            self.progressbar.set_fraction(0.0)

    def on_right_to_left_toggled(self, button):
        value = button.get_active()
        self.progressbar.set_inverted(value)

    # This is unused code from an example (?)
    def on_timeout(self, user_data):
        """
        Update value on the progress bar
        """
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            new_value = self.progressbar.get_fraction() + 0.01

            if new_value > 1:
                new_value = 0

            self.progressbar.set_fraction(new_value)

        # As this is a timeout function, return True so that it
        # continues to get called
        return True


if __name__ == "__main__":
    import pyglet
    window = pyglet.window.Window()
    label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

    @window.event
    def on_draw():
        window.clear()
        label.draw()

    pyglet.app.run()
