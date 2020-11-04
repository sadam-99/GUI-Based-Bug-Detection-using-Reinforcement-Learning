import os, json, sys, time
import cv2 as cv
import numpy as np
from sklearn.utils import shuffle

class DatasetBuilder:

    def __init__(self, base_path="C:\\Users\\Wafid\\Downloads\\traces\\filtered_traces\\", image_dims=(96,54), bw=True, n_apps=3000, n_previous_states=0):

        self.base_path = base_path
        self.image_dims = image_dims
        self.n_apps = n_apps
        self.bw = bw
        self.train_data = []
        self.train_label = []
        self.n_previous_states = n_previous_states


    def build(self):
        apps = os.listdir(self.base_path)
        i = 0
        for app in apps[:self.n_apps]:
            i += 1

            app_path = os.path.join(self.base_path, app)

            traces = os.listdir(app_path)

            for trace in traces:
                trace_path = os.path.join(app_path, trace)

                json_file = open(trace_path + "\\gestures.json")
                json_data = json.load(json_file)

                screen_path = os.path.join(trace_path, 'screenshots')
                images = os.listdir(screen_path)

                start_index = int(len(images) / 2)

                for img_name in images[start_index:]:

                    coordinates = json_data[img_name.split(".")[0]]
                    coord_len = len(coordinates) # if > 1, it a swipe, otherwise, its a tap

                    if coord_len == 0:
                        continue

                    if self.bw:
                        img = cv.imread(screen_path + '\\' + img_name,0)
                    else:
                        img = cv.imread(screen_path + '\\' + img_name)

                    # img = cv.resize(img, (self.image_dims[1],self.image_dims[0]))
                    img = cv.resize(img, (self.image_dims[0],self.image_dims[1]))


                    self.train_data.append(img)

                    if coord_len == 1:
                        self.train_label.append([coordinates[0][0], coordinates[0][1], coordinates[0][0], coordinates[0][1] ] )
                    else:
                        self.train_label.append([coordinates[0][0], coordinates[0][1], coordinates[coord_len-1][0], coordinates[coord_len-1][1] ] )

                json_file.close()

            if i%50 == 0:
                os.system('cls')
                print('Loaded %d apps' %  i)


        self.train_data = np.asarray(self.train_data, dtype='float32')
        self.train_label = np.asarray(self.train_label)

        if self.bw:
            self.train_data = self.train_data.reshape(self.train_data.shape[0], self.image_dims[0], self.image_dims[1], 1) # reshaping to a 4D tensor with the desired dimensions

        self.train_data, self.train_label = shuffle(self.train_data, self.train_label, random_state=1)
        print("Training dataset shape: ")
        print(self.train_data.shape)
        print("Training labels shape: ")
        print(self.train_label.shape)

        time.sleep(1)

        return self.train_data, self.train_label

    def build_stacked(self):
        apps = os.listdir(self.base_path)
        app_counter = 0
        for app in apps[:self.n_apps]:
            app_counter += 1

            app_path = os.path.join(self.base_path, app)

            traces = os.listdir(app_path)

            for trace in traces:
                trace_path = os.path.join(app_path, trace)

                json_file = open(trace_path + "\\gestures.json")
                json_data = json.load(json_file)

                screen_path = os.path.join(trace_path, 'screenshots')
                images = os.listdir(screen_path)


                start_index = int(len(images) / 2)


                for i in range(len(images[start_index:])):
                    input_data = []
                    for j in range(i,i+self.n_previous_states):

                        img_index = start_index + j
                        if img_index >= len(images):
                            img_index = len(images) - 1


                        coordinates = json_data[images[img_index].split(".")[0]]
                        coord_len = len(coordinates) # if > 1, it a swipe, otherwise, its a tap

                        if coord_len == 0:
                            continue

                        if self.bw:
                            img = cv.imread(screen_path + '\\' + images[img_index],0)
                        else:
                            img = cv.imread(screen_path + '\\' + images[img_index])

                        # img = cv.resize(img, (self.image_dims[1],self.image_dims[0]))
                        img = cv.resize(img, (self.image_dims[0],self.image_dims[1]))

                        input_data.append(img)

                        if len(input_data) == self.n_previous_states:
                            if coord_len == 1 :
                                self.train_label.append([coordinates[0][0], coordinates[0][1], coordinates[0][0], coordinates[0][1] ] )
                            else:
                                self.train_label.append([coordinates[0][0], coordinates[0][1], coordinates[coord_len-1][0], coordinates[coord_len-1][1] ] )

                    if len(input_data) == self.n_previous_states:
                        self.train_data.append(np.array(input_data))

                json_file.close()

            if app_counter%50 == 0:
                os.system('cls')
                print('Loaded %d apps' %  app_counter)


        self.train_data = np.array(self.train_data, dtype='float32') / 255.0
        self.train_label = np.array(self.train_label)

        if self.bw:
            self.train_data = self.train_data.reshape(self.train_data.shape[0], self.n_previous_states, self.image_dims[0], self.image_dims[1], 1) # reshaping to a 4D tensor with the desired dimensions

        self.train_data, self.train_label = shuffle(self.train_data, self.train_label, random_state=1)
        print("Training dataset shape: ")
        print(self.train_data.shape)
        print("Training labels shape: ")
        print(self.train_label.shape)

        time.sleep(1)

        return self.train_data, self.train_label
