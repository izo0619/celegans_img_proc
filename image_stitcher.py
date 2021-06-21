from PIL import Image

# params
source_dir = 'source_images/'
dest_dir = 'stiched_images/'
name = "cat"
ext = ".jpeg"
rows = 2
cols = 2
row_imgs = []

# helper functions
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

# probably one more for loop around here to go through each set of pics
# for img_set in 
for i in range(rows):
    # set first image (image on the far left of the row)
    base_img = Image.open(source_dir + name  +'-' + str(i*cols) + ext)
    # loop through each column and add respective image
    for j in range(1,cols):
        concat_img = Image.open(source_dir + name +'-' + str((i*cols) + j) + ext)
        base_img = get_concat_h(base_img, concat_img)
    # keep each row img in an array of img objects to concat into final prod
    row_imgs.append(base_img)

# stich each stiched row together to form final image
final_img = row_imgs[0]
for i in range(1,rows):
    final_img = get_concat_v(final_img, row_imgs[i])

final_img.save(dest_dir + name + "-stiched.jpg")