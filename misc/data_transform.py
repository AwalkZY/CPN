import json
import h5py
import os
import numpy as np
import csv


def load_json(filename):
    with open(filename, "r") as json_file:
        return json.load(json_file)


def save_json(content, filename):
    with open(filename, "w") as json_file:
        json.dump(content, json_file)


def vsl_transform(action="train"):
    result = []
    total_json = load_json("charades.json")
    for text_line in open("charades_sta_{}.txt".format(action)):
        """DM2HW 10.0 17.9##person takes a box."""
        items = text_line.split("##")
        items = (*(items[0].split(" ")), items[-1])
        name, start_time, end_time, desc = items
        start_time, end_time = float(start_time), float(end_time)
        info = total_json[name]
        if info["subset"] != action + "ing":
            continue
        duration = info["duration"]
        start_time, end_time = max(0.0, start_time), min(end_time, duration)
        # if start_time > info["duration"] or start_time > end_time or end_time > info["duration"]:
        #     continue
        result.append([name, duration, [start_time, end_time], desc])
    save_json(result, action + ".json")


def lgi_transform(action="train"):
    duration_dict = {}
    result = []
    with open('Charades_v1_{}.csv'.format(action)) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row[0]) == 5:
                duration_dict[row[0]] = float(row[-1])
    for text_line in open("charades_sta_{}.txt".format(action)):
        """DM2HW 10.0 17.9##person takes a box."""
        items = text_line.split("##")
        items = (*(items[0].split(" ")), items[-1])
        name, start_time, end_time, desc = items
        start_time, end_time = float(start_time), float(end_time)
        duration = duration_dict[name]
        if start_time > end_time or start_time > duration or end_time > duration:
            continue
        result.append([name, duration, [start_time, end_time], desc])
    save_json(result, action + ".json")


def extract_2dtan(filename, dir_name):
    file = h5py.File(filename, "r")
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for feature_name in file.keys():
        new_file_name = feature_name.split(".")[0] + ".npy"
        np.save(os.path.join(dir_name, new_file_name), file[feature_name])


def tacos_transform(action="train"):
    results = []
    total_json = load_json(action + ".json")
    for name in total_json.keys():
        times = total_json[name]["timestamps"]
        descs = total_json[name]["sentences"]
        duration = int(total_json[name]["num_frames"] / total_json[name]["fps"] * 100) / 100.0
        for time, desc in zip(times, descs):
            start_time = int(time[0] / total_json[name]["fps"] * 100) / 100.0
            end_time = int(time[1] / total_json[name]["fps"] * 100) / 100.0
            results.append([name, duration, [start_time, end_time], desc])
    save_json(results, action + "_data.json")


tacos_transform("train")
tacos_transform("test")
