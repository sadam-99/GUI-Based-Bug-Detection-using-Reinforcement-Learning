import os
import sys
import glob
import json

# This is the directory for the apps folder
apps_path = r"Apps"

# function for loading the json files#
def load_json(json_path):
    with open(json_path, 'r') as f:
        json_doc = json.load(f)
        return json_doc



apps_dir = r'Apps\*\*\view_hierarchies'
json_files = glob.glob(apps_dir + '\*.json')
json_files
c= 0
for i in range(0,len(json_files)):
    jso = load_json(json_files[i])
    boxes_1 = jso['activity']['root']['children'][0]['bounds']
    boxes_2 = jso['activity']['root']['children'][0]['children'][0]['bounds']
    boxes_3 = jso['activity']['root']['children'][0]['children'][1]['bounds']
    
    c+=1
    print("The Bounding boxes for the file are:", json_files[i].split('\\')[-1],"\n", boxes_1, boxes_2, boxes_3)
