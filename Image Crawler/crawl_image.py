import urllib
import random
import requests
import os

n_image = 20

width = 600
height = 400
size = str(width)+"x"+str(height)
pitch = -0.76

eps = 0.00005

#coordinates
min_lat = 52.29
max_lat = 52.42
min_long = 4.73
max_long = 4.98

def generate_random_point():
	rdm_lat = random.uniform(min_lat, max_lat)
	rdm_long = random.uniform(min_long, max_long)
	return [rdm_lat,rdm_long]

	
def get_heading(lat,long):
	d_lat = lat + 10*eps
	d_long = long
	found = 0
	head = 90
	while(not(found) and head<180):
		if(image_valid(d_lat,d_long)):
			return head
		d_lat = d_lat - eps
		d_long = d_long + eps
		head = head + 9
	
	return 0
	
def image_valid(lat,long):
	location = str(lat)+","+str(long)
	heading = 0
	url = "https://maps.googleapis.com/maps/api/streetview?size="+size+"&location="+location+"&heading="+str(heading)+"&pitch="+str(pitch)
	urllib.urlretrieve(url, "test.jpg")
	statinfo = os.stat("test.jpg")
	if(statinfo.st_size < 7000):
		return 0
	return 1

def download_image(lat,long,heading,filename):
	location = str(lat)+","+str(long)
	url = "https://maps.googleapis.com/maps/api/streetview?size="+size+"&location="+location+"&heading="+str(heading)+"&pitch="+str(pitch)
	urllib.urlretrieve(url, filename)
	
def process_location(lat,long,it):
	if(not(image_valid(lat,long))):
		return 0
	location = str(lat)+","+str(long)
	heading = get_heading(lat,long)
	
	for k in range(1,5):
		head = heading + 90*k
		filename = str(it)+"_"+str(k)+".jpg"
		download_image(lat,long,head,filename)
	
	print(location+" heading:"+str(heading))
	return 1
	
def start_crawling():
	for iter in range(1,n_image+1):
		valid = 0
		while(not(valid)):
			random_point = generate_random_point()
			valid = process_location(random_point[0],random_point[1],iter)
			

start_crawling()