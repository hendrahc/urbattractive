import json

input_filename = "../../../Dataset/dataset_PILOT.txt"
output_filename = "../Locator/coordinates_PILOT.js"
input_file = open(input_filename,"r")
output_file = open(output_filename,"w")

dat = []
names = []

for line in input_file:
    fields = line.split(";")
    if(len(fields) == 5):
        if(fields[0] not in names):
            rec = {}
            rec["name"] = fields[0]
            rec["lat"] = fields[2]
            rec["long"] = fields[3]
            dat.append(rec)
            names.append(rec["name"])

dat_json = json.JSONEncoder().encode(dat)
output_file.write("var coordinates_data = "+dat_json)
input_file.close()
output_file.close()