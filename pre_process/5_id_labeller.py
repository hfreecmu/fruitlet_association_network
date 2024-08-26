import os
import numpy as np
import json
import pickle
import tkinter
from PIL import Image, ImageDraw, ImageTk 

circule_radius = 2

resize = False
resize_scale = 1.8

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(path, data, pretty=False):
    with open(path, 'w') as f:
        if not pretty:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=4)

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

class Annotate():
    def __init__(self, image_dir, detections_dir, output_dir, year, side):
        self.image_dir = image_dir
        self.detections_dir = detections_dir
        self.output_dir = output_dir
        self.year = year
        self.side = side

        self.should_quit = None
        self.should_delete = None
        self.should_save = None
        self.curr_index = None
        self.prev_index = None
        self.num_files = None

        self.curr_annotation_dict = None

        self.label_mode = None
        self.fruitlet_num = None

    def annotate(self):
        # set up the gui
        window = tkinter.Tk()
        window.bind("<Key>", self.event_action)
        window.bind("<Button-1>", self.event_action_click)

        self.files = self.get_files(self.year, self.side)
        self.files = sorted(self.files)

        self.should_quit = False
        self.curr_index = 0
        self.prev_index = None
        self.num_files = len(self.files)
        self.label_mode = False

        while not self.should_quit:
            self.should_save = False
            self.should_delete = False

            file_key = self.files[self.curr_index]

            image_path = os.path.join(self.image_dir, file_key)
            detections_path = os.path.join(self.detections_dir, file_key.replace('.png', '.pkl'))

            boxes = read_pickle(detections_path)['boxes']

            annotation_filename = file_key.replace('.png', '.json')
            annotation_path = os.path.join(self.output_dir, annotation_filename)

            if self.curr_index == self.prev_index:
                pass
            elif os.path.exists(annotation_path):
                self.curr_annotation_dict = read_json(annotation_path)
            else:
                self.curr_annotation_dict = dict()
                
                self.curr_annotation_dict['image_path'] = image_path
                self.curr_annotation_dict['det_path'] = detections_path

                self.curr_annotation_dict['annotations'] = []

                for box in boxes:
                    x0, y0, x1, y1, score = box
                    entry = {'x0': x0,
                             'y0': y0,
                             'x1': x1,
                             'y1': y1,
                             'score': score,
                             'fruitlet_id': -1}
                    self.curr_annotation_dict['annotations'].append(entry)

            self.prev_index = self.curr_index

            window.title(file_key)
            picture = Image.open(image_path)
            if resize:
                raise RuntimeError('resize not supported yet')
            
            picture_draw = ImageDraw.Draw(picture)

            is_valid = True
            valid_set = set()
            label_infos = []
            for box in self.curr_annotation_dict['annotations']:
                x0 = int(box['x0'])
                y0 = int(box['y0'])
                x1 = int(box['x1'])
                y1 = int(box['y1'])

                pts = [(x0, y0), (x1, y1)]

                mid_x = int((x0 + x1)/2)
                mid_y = int((y0 + y1)/2)

                if box['fruitlet_id'] >= 0:
                    color = 'cyan'
                    label_infos.append([mid_x, mid_y, box['fruitlet_id']])
                else:
                    color = "red"

                picture_draw.rectangle(pts, outline=color)

                if (box['fruitlet_id'] >= 0) and (box["fruitlet_id"] in valid_set):
                    is_valid = False

                valid_set.add(box["fruitlet_id"])

                picture_draw.ellipse([(mid_x - circule_radius, mid_y - circule_radius), (mid_x + circule_radius, mid_y + circule_radius)], fill='purple')
                    
            tk_picture = ImageTk.PhotoImage(picture)
            picture_width = picture.size[0]
            picture_height = picture.size[1]
            window.geometry("{}x{}+100+100".format(picture_width, picture_height))
            image_widget = tkinter.Label(window, image=tk_picture)
            image_widget.place(x=0, y=0, width=picture_width, height=picture_height)

            label_id = file_key.replace('.png', '')
            if is_valid:
                label_string = 'valid ' + str(label_id)
                label_colour = 'green'
            else:
                label_string = 'invalid ' + str(label_id)
                label_colour = 'red'

            label_text = tkinter.Label(window, text=label_string, font=("Helvetica", 22), fg=label_colour)
            label_text.place(anchor = tkinter.NW, x = 0, y = 0)

            for label_info in label_infos:
                mid_x, mid_y, assoc_id = label_info
                num_text = tkinter.Label(window, text=str(assoc_id), font=("Helvetica", 8))
                num_text.place(x=mid_x, y=mid_y+10)

            # wait for events
            window.mainloop()

            if self.should_quit:
                continue

            assert not (self.should_save and self.should_delete)

            if self.should_save:
                write_json(annotation_path, self.curr_annotation_dict, pretty=True)

            if self.should_delete:
                if os.path.exists(annotation_path):
                    os.remove(annotation_path)

    def event_action(self, event):
        character = event.char
        
        if character == 'q':
            self.should_quit = True
            event.widget.quit()
        elif character == 's':
            self.should_save = True
            self.should_delete = False
            self.prev_index = None
            event.widget.quit()
        elif character == 'b':
            self.should_save = False
            self.should_delete = True
            self.prev_index = None
            event.widget.quit()
        elif character in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            self.label_mode = True
            self.fruitlet_num = int(character)
        elif character == 'c':
            self.label_mode = False
            self.fruitlet_num = None
        elif character == 'k':
            self.label_mode = False
            self.fruitlet_num = None
            self.prev_index = None
            event.widget.quit()
        elif character == 'a':
            if self.curr_index > 0:
                self.curr_index -= 1
                event.widget.quit()
        elif character == 'd':
            if self.curr_index < self.num_files - 1:
                self.curr_index += 1
                event.widget.quit()
        elif character == 'e':
            year, tag_id, _, _ = self.files[self.curr_index].split('_')
            identifier = '_'.join([year, tag_id, '']) #should end with _

            prev_index = self.curr_index
            prev_identifier = identifier
            while (prev_index > 0) and (prev_identifier == identifier):
                prev_index -= 1
                year, tag_id, _, _ = self.files[prev_index].split('_')
                prev_identifier = '_'.join([year, tag_id, ''])

            if prev_index != self.curr_index:
                self.curr_index = prev_index
                event.widget.quit()
        elif character == 'r':
            year, tag_id, _, _ = self.files[self.curr_index].split('_')
            identifier = '_'.join([year, tag_id, '']) #should end with _

            next_index = self.curr_index
            next_identifier = identifier
            while (next_index < self.num_files - 1) and (next_identifier == identifier):
                next_index += 1
                year, tag_id, _, _ = self.files[next_index].split('_')
                next_identifier = '_'.join([year, tag_id, ''])

            if next_index != self.curr_index:
                self.curr_index = next_index
                event.widget.quit()
               

    def event_action_click(self, event):
        if not self.label_mode:
            return

        x = event.x
        y = event.y
   
        min_dist = None
        min_box = None
        for box in self.curr_annotation_dict["annotations"]:
            x0 = box['x0']
            y0 = box['y0']
            x1 = box['x1']
            y1 = box['y1']

            mid_x = (x0 + x1)/2
            mid_y = (y0 + y1)/2

            dist = np.square(mid_x - x) + np.square(mid_y - y)
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                min_box = box

        if min_box is not None:
            min_box['fruitlet_id'] = self.fruitlet_num

        self.label_mode = False
        self.fruitlet_num = None

        event.widget.quit()

    def get_files(self, year, side):
        files = []
        for filename in os.listdir(self.image_dir):
            if year is not None:
                if not filename.split('_')[0] == str(year):
                    continue                    

                #temporarily hardcoding
                if int(filename.split('_')[1]) < 31:
                    continue

            if side is not None:
                if not side in filename:
                    continue

            files.append(filename)

        return files

image_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images/images'
detections_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/detections'
output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations'
year = 2023
side = 'left'

if __name__ == "__main__":
    annotate = Annotate(image_dir, detections_dir, output_dir, year, side)
    annotate.annotate()