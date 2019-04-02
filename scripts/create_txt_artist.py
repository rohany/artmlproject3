import os, json


for folder in os.listdir():
    if os.path.isdir(folder):
        with open(folder + ".txt", "w") as txt_file:
            for file in os.listdir(folder):
                with open(folder + "/" + file) as json_file:
                    json_data = json.load(json_file)
                    lyrics = json_data['songs'][0]['lyrics']
                    txt_file.write(lyrics)