import glob
import os

input_loc = "../../Dataset/PILOT/"
input_filename = "../../Dataset/log_PILOT.txt"
output_filename = "../../Dataset/dataset_PILOT.txt"

input_file = open(input_filename,"r")
output_file = open(output_filename,"w")


#list image files
os.chdir(input_loc)
images = glob.glob("*.jpg")
for line in input_file:
    fields = line.split(";")
    if (len(fields) == 5):
        im_name = fields[1]
        if(im_name in images):
            output_file.write(line)
            images.remove(im_name)

if(len(images)>0):
    print(images)

input_file.close()
output_file.close()