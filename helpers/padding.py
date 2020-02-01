#########################################################
# resize.py
#
# Simple script using only PIL to resize an image.
# 
# By M. Harris, 2018
#########################################################

from PIL import Image
import argparse
import glob

# Collect command line arguments
parser = argparse.ArgumentParser(description='Process command line args.')
parser.add_argument('--input_img', type=str, 
                    help='Input image path')
parser.add_argument('--height', type=int,
                    help='New height in pixels')
parser.add_argument('--out', type=str,
                    help='Output image path')
args = parser.parse_args()


baseheight = args.height
basewidth = args.height
img_output = str(args.out)
# Create the for loop here
input_dir = args.input_img
image_list = [f for f in glob.glob(input_dir+"/*.jpg")]
for file in image_list:
    img = Image.open(file)
    filename = str(file).rsplit("/", 1)[-1]
    print("File name is: "+filename)
    hpercent = (baseheight / float(img.size[1]))
    wpercent = (basewidth / float(img.size[0]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((baseheight, baseheight), Image.ANTIALIAS)
    img.save(img_output+"/"+filename)
