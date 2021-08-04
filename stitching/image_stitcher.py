from PIL import Image
import os
import glob


# params
root_path = '/Users/isabelzhong/Box/10x_wormimages'
directory_contents = os.listdir(root_path)
ext = ".TIF"
rows = 5
cols = 5
row_imgs = []
well_alphas = [chr(i) for i in range(ord('a'),ord('f')+1)]
error_wells = {}

# helper functions
# horizontally concatenates two images
def get_concat_h(im1, im2):
    dst = Image.new('I;16', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# vertically concatenates two images
def get_concat_v(im1, im2):
    dst = Image.new('I;16', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

# stitches one well together
# params: subdir - the subdirectory that contains the folder where the set of images lies in
#         well - the letter of the well

# this worked on a file flow like so:
# - 10x_wormimages (root path)
#     - p01-growth-H01-10X (sub directory)
#         - TimePoint_1
#             - p01-growth-H01-10X_A01_s1.tif
#             - p01-growth-H01-10X_A01_s2.tif
#             - ...
#     - p02-growth-H02-10X (sub directory)
#     - ...
def stitch_well(subdir, well):
    if os.path.isdir(os.path.join(root_path, subdir)):
        dest_dir = root_path + '/' + subdir + '/stitched_images/'
        # creates a folder called stitched_images if it doesn't exist already
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        # set of images within the subdirectory that we are trying to stitch
        source_dir = os.path.join(root_path, subdir, 'TimePoint_1/')
        
        name_header = subdir[:19]
        try:
            print('running well ' + well + ' in subdir ' + subdir)
            row_imgs = []
            # creates array of stitched rows of images
            for i in range(rows):
                # set first image (image on the far left of the row)
                base_img = Image.open(glob.glob(source_dir + name_header + well.upper() + '0' + '?' + '_s' + str(i*cols+1) + ext)[0])
                # loop through each column and add respective image
                for j in range(1,cols):
                    concat_img = Image.open(glob.glob(source_dir + name_header + well.upper() + '0' + '?' + '_s' + str((i*cols) + j + 1) + ext)[0])
                    base_img = get_concat_h(base_img, concat_img)
                # keep each row img in an array of img objects to concat into final prod
                row_imgs.append(base_img)

            # stitch each stitched row together to form final image
            final_img = row_imgs[0]
            for i in range(1,len(row_imgs)):
                final_img = get_concat_v(final_img, row_imgs[i])

            final_img.save(dest_dir + 'well-' + well + "-stitched" + ext)
        except Exception as e:
            print("error while running well " + well + " in hour " + subdir + '\n')
            print(e)

# stitches all wells in the directory together
def stitch_all_in_dir():
    error_wells = {}
    for subdir in directory_contents:
        if os.path.isdir(os.path.join(root_path, subdir)):
            dest_dir = root_path + '/' + subdir + '/stitched_images/'
            if not os.path.exists(dest_dir):
                os.mkdir(dest_dir)

            source_dir = os.path.join(root_path, subdir, 'TimePoint_1/')
                
            name_header = subdir[:19]

            for well in well_alphas:
                print('running well ' + well + ' in subdir ' + subdir + '\n')
                try:
                    row_imgs = []
                    for i in range(rows):
                        # set first image (image on the far left of the row)
                        base_img = Image.open(source_dir + name_header + '_' + well + '01_s' + str(i*cols+1) + ext)
                        # loop through each column and add respective image
                        for j in range(1,cols):
                            concat_img = Image.open(source_dir + name_header + '_' + well + '01_s' + str((i*cols) + j + 1) + ext)
                            base_img = get_concat_h(base_img, concat_img)
                        # keep each row img in an array of img objects to concat into final prod
                        row_imgs.append(base_img)

                    # stitch each stitched row together to form final image
                    final_img = row_imgs[0]
                    for i in range(1,len(row_imgs)):
                        final_img = get_concat_v(final_img, row_imgs[i])

                    final_img.save(dest_dir + 'well-' + well + "-stitched" + ext)
                except Exception as e:
                    print("error while running well " + well + " in hour " + subdir)
                    print(e)
                    # store all error wells in a dictionary, should look like this but haven't really tested this part
                    # too much:
                        # error_wells = {
                        #     'p42-growth-H42-10X_Plate_3275': ['d', 'e', 'f'],
                        #     'p43-growth-H43-10X_Plate_3277': ['e', 'f']}
                    error_wells.setdefault(subdir, []).append(well)


# runs wells with errors
def stitch_error_wells():
    for sd, w in error_wells.items():
        for well in w:
            stitch_well(sd,well)
