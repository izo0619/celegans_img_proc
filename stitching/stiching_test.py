from PIL import Image
import os
import glob


# params
root_path = '/Users/isabelzhong/Box/10x_wormimages'
directory_contents = os.listdir(root_path)
ext = ".tif"
rows = 5
cols = 5
row_imgs = []
well_alphas = [chr(i) for i in range(ord('a'),ord('f')+1)]
error_wells = {}

# helper functions
def get_concat_h(im1, im2):
    dst = Image.new('I', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('I', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def stich_well(subdir, well):
    if os.path.isdir(os.path.join(root_path, subdir)):
        dest_dir = root_path + '/' + subdir + '/stiched_images/'
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        source_dir = os.path.join(root_path, subdir, 'TimePoint_1/')
            
        name_header = subdir[:19]

        try:
            print('running well ' + well + ' in subdir ' + subdir)
            row_imgs = []
            for i in range(rows):
                # set first image (image on the far left of the row)
                # test_dir = glob.glob(source_dir + name_header + well + '0' + '?' + '_s' + str(i*cols+1) + ext)
                # print(test_dir)
                # for file in test_dir:
                #     print(file)
                # base_img = Image.open(source_dir + name_header + well + '03_s' + str(i*cols+1) + ext)
                base_img = Image.open(glob.glob(source_dir + name_header + well.upper() + '0' + '?' + '_s' + str(i*cols+1) + ext)[0])
                # loop through each column and add respective image
                for j in range(1,cols):
                    concat_img = Image.open(glob.glob(source_dir + name_header + well.upper() + '0' + '?' + '_s' + str((i*cols) + j + 1) + ext)[0])
                    base_img = get_concat_h(base_img, concat_img)
                # keep each row img in an array of img objects to concat into final prod
                row_imgs.append(base_img)

            # stich each stiched row together to form final image
            final_img = row_imgs[0]
            for i in range(1,len(row_imgs)):
                final_img = get_concat_v(final_img, row_imgs[i])

            final_img.save(dest_dir + 'well-' + well + "-stiched" + ext)
        except Exception as e:
            print("error while running well " + well + " in hour " + subdir + '\n')
            print(e)


def stich_all_in_dir():
    for subdir in directory_contents:
        if os.path.isdir(os.path.join(root_path, subdir)):
            dest_dir = root_path + '/' + subdir + '/stiched_images/'
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
                        base_img = Image.open(source_dir + name_header + well + '01_s' + str(i*cols+1) + ext)
                        # loop through each column and add respective image
                        for j in range(1,cols):
                            concat_img = Image.open(source_dir + name_header + well + '01_s' + str((i*cols) + j + 1) + ext)
                            base_img = get_concat_h(base_img, concat_img)
                        # keep each row img in an array of img objects to concat into final prod
                        row_imgs.append(base_img)

                    # stich each stiched row together to form final image
                    final_img = row_imgs[0]
                    for i in range(1,len(row_imgs)):
                        final_img = get_concat_v(final_img, row_imgs[i])

                    final_img.save(dest_dir + 'well-' + well + "-stiched" + ext)
                except Exception as e:
                    print("error while running well " + well + " in hour" + subdir)
                    print(e)
                    error_wells.setdefault(subdir, []).append(well)




dest_dir = root_path + '/' + 'p04-growth-H04-10X/'

source_dir = root_path + '/' + 'p04-growth-H04-10X' + '/Well_A01_Corrected/'
    
name_header = 'p04-growth-H04-10X_A01_s'

try:
    row_imgs = []
    for i in range(rows):
        base_img = Image.open(source_dir + name_header + str(i*cols+1) + ext)
        
        # loop through each column and add respective image
        for j in range(1,cols):
            print(source_dir + name_header + str((i*cols) + j + 1) + ext)
            concat_img = Image.open(source_dir + name_header + str((i*cols) + j + 1) + ext)
            base_img = get_concat_h(base_img, concat_img)
        # keep each row img in an array of img objects to concat into final prod
        row_imgs.append(base_img)

    # stich each stiched row together to form final image
    final_img = row_imgs[0]
    for i in range(1,len(row_imgs)):
        final_img = get_concat_v(final_img, row_imgs[i])

    final_img.save(dest_dir + 'h4-well-a-basic-corrected-stiched' + ext)
except Exception as e:
    print("error while running well")
    print(e)