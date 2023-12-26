import os
import argparse

'''
This file is used to generate a txt file that contains the label and full path of the input images. These filenames
are later used in the dctp model to generate labels and images. This script should be placed in the same directory as the
root directory of the training images.

How to use:
Pass in the path to the root directory of the training images and the name of the output file
'''

# iterate over all the files and subdirs in the provided imgs dir and grab the filenames
def get_all_gpgs(subdir):
	target_dir = os.path.join(subdir)
	label_translation = []

	f = []
	w = os.walk(target_dir)
	for (dirpath, dirnames, filenames) in w:
		if 'jpg' in '\t'.join(filenames):
			for file in filenames:
				image_subdir = dirpath.replace(subdir,'')
				label = image_subdir.split('/')[1]
				f.append(str(label) + ' ' + os.path.join(dirpath, file))

	return f

# set the incoming args
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate an index file containing image file paths')
    parser.add_argument('--output-file', help='name of the generated index file', required=True)
    parser.add_argument('--input_root_dir', help='root directory of the images folder', required=True)

    args = parser.parse_args()
    return args


# parse the input args
args = parse_arguments()

print('Looking at imgs dir: ', args.input_root_dir)
print('Printing to file: ', args.output_file)

# list out all the files in Indian and write to file called output.txt
target_dir = args.input_root_dir
files = []
files += get_all_gpgs(target_dir)

# print out to the desired file
out_txt = '\n'.join(files)
f = open(args.output_file, "a")
f.write(out_txt)
f.close()
