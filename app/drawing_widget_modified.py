# Original code from: ipycanvas-drawing 0.0.5
# Author: rubenw rubenwiersma@gmail.com
# Changes by Beñat Berasategui

from ipycanvas import MultiCanvas, hold_canvas
from ipywidgets import IntSlider, Button, Label, HBox, VBox, link, Layout, Image, Dropdown, Output, AppLayout
import numpy as np

class DrawingWidget:
    drawing = False
    position = None
    shape = []
    output_array = None
    drawing_line_width = 15
    history = []
    future = []
    max_history = 100
    
    def __init__(self, width, height, external_function, background="#f5f5f5", alpha=1.0, default_radius=15):
        self.background = background
        self.alpha = alpha
        self.default_radius = default_radius
        self.init_canvas(width, height)
        self.canvas.on_client_ready(self.on_client_ready)
        self.external_function = external_function
        # if not pred_img: 
        #     self.pred_img = Image(value=open("png/Delta.png", "rb").read(),format='png') #,width=150,height=150,)
        #     self.pred_img.layout.margin = '20px 0 0 50px' #top/right/bottom/left
        #     self.pred_img.layout.height='100px'
        #     self.pred_img.layout.object_fit = 'contain'
        #     # self.pred_img.layout.border = 'solid 1px'
        # else:
        #     self.pred_img = pred_img
        
        output_dropdown = Output()
        #output_dropdown.layout.margin = '0 0 0 50px' #top/right/bottom/left
        self.output_dropdown = output_dropdown
        
        output_img = Output()
        #output_img.layout.margin = '0 0 0 50px'
        self.output_img = output_img

    def get_image_data(self, background=False):
        #if background:
            #return self.canvas.get_image_data()
        #return self.canvas._canvases[1].get_image_data()
        return self.canvas[1].get_image_data()

    def init_canvas(self, width, height):
        self.canvas = MultiCanvas(n_canvases=3, width=width, height=height, sync_image_data=True)
        self.canvas._canvases[1].sync_image_data = True
        
        self.canvas.on_mouse_down(self.on_mouse_down)
        self.canvas.on_mouse_move(self.on_mouse_move)
        self.canvas.on_mouse_up(self.on_mouse_up)
        

    def reset_background(self):
        with hold_canvas():
            if isinstance(self.background, np.ndarray):
                self.canvas[0].put_image_data(self.background)
            else:
                self.canvas[0].fill_style = self.background
                self.canvas[0].fill_rect(0, 0, self.canvas.width, self.canvas.height)

    def on_client_ready(self):
        self.reset_background()
        self.set_canvas_properties()
        #self.draw_initial_point()

    def set_canvas_properties(self):
        self.canvas[2].stroke_style = "#4287F5"
        self.canvas[2].fill_style = "#4287F5"
        self.canvas[2].line_cap = 'round'
        self.canvas[2].line_width = self.drawing_line_width
        
        self.canvas[1].line_cap = 'round'
        self.canvas[1].line_join = 'round'
        self.canvas[1].line_width = self.default_radius
        self.canvas[1].global_alpha = self.alpha
        
    def call_external_function(self, *args):
        self.external_function()

    def show(self):
        # Icons from: https://fontawesome.com/search
        radius_slider = IntSlider(value=self.default_radius, min=1, max=30)
        reset_button = Button(description="Restaurar brocha")
        reset_button.on_click(self.reset_slider_value)
        clear_button = Button(description="Borrar", icon="eraser")
        clear_button.on_click(self.clear_canvas)
        undo_button = Button(description="Deshacer", icon="rotate-left")
        undo_button.on_click(self.undo)
        redo_button = Button(description="Rehacer", icon="rotate-right")
        redo_button.on_click(self.redo)
        example_button = Button(description="Ejemplo", icon="question")
        example_button.on_click(self.draw_initial_point)
        predict_button = Button(description="Predecir", icon="eye")
        predict_button.on_click(self.call_external_function)
        text = Label(value="Tamaño de la brocha", style={'font_size':'13pt', 'font_weight':'bold'})
        text.layout.margin = '20px 5px 0 0'
        #space = Layout(flex='2')
        self.canvas.layout.margin = '0 50px 0 0'

        link((radius_slider, "value"), (self.canvas[1], "line_width"))
        
        
        dropdown_title = Label(value="Predicción", style={'font_size':'13pt', 'font_weight':'bold'})
        #dropdown_title.layout.margin = '0px 0 0 50px'
#         dropdown = Dropdown(
#                             options=[(r'\Delta', 1), (r'\triangle', 2), (r'\vartriangle', 3)],
#                             value=1,
#                             #description='Number:',
#                         )
        
#         dropdown.layout.margin = '0 0 0 50px' #top/right/bottom/left

        buttons = HBox(( VBox((clear_button, undo_button, redo_button)), VBox((example_button, predict_button, reset_button)) ))
        center = HBox((self.canvas, VBox((buttons, text, radius_slider)) ))
        right = VBox((dropdown_title, self.output_dropdown, self.output_img)) 
        return AppLayout(center=center, right_sidebar=right, pane_widths=[0, 7, 5])
    
        # return HBox((self.canvas, VBox((buttons, text, radius_slider)), VBox((dropdown_title, self.output_dropdown, self.output_img)) )) 

    def save_to_history(self):
        self.history.append(self.canvas._canvases[1].get_image_data())
        if len(self.history) > self.max_history:
            self.history = self.history[1:]
        self.future = []

    def on_mouse_down(self, x, y):
        self.drawing = True
        self.position = (x, y)
        self.shape = [self.position]
        self.save_to_history()

    def on_mouse_move(self, x, y):
        if not self.drawing:
            return

        with hold_canvas():
            self.canvas[2].line_width = self.canvas[1].line_width
            self.canvas[2].stroke_line(self.position[0], self.position[1], x, y)
            self.canvas[2].line_width = self.drawing_line_width

            self.position = (x, y)
            
        self.shape.append(self.position)

    def on_mouse_up(self, x, y):
        self.drawing = False

        with hold_canvas():
            self.canvas[2].clear()
            self.canvas[1].stroke_lines(self.shape)

        self.shape = []

    def clear_canvas(self, *args):
        self.save_to_history()
        with hold_canvas():
            self.canvas[1].clear()

    def undo(self, *args):
        if self.history:
            with hold_canvas():
                self.future.append(self.canvas._canvases[1].get_image_data())
                self.canvas[1].clear()
                self.canvas[1].put_image_data(self.history[-1])
                self.history = self.history[:-1]

    def redo(self, *args):
        if self.future:
            with hold_canvas():
                self.history.append(self.canvas._canvases[1].get_image_data())
                self.canvas[1].clear()
                self.canvas[1].put_image_data(self.future[-1])
                self.future = self.future[:-1]

    def draw_initial_point(self, *args):
        with hold_canvas():
            # Load initial drawing
            loaded_array = np.load("initial_drawing.npy")
            self.canvas[1].clear()
            self.canvas[1].put_image_data(loaded_array)
            
    def reset_slider_value(self, *args):
        self.canvas[1].line_width = self.default_radius
