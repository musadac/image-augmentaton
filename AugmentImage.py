import argparse
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw
import random
import numpy as np
import os
import cv2

parser = argparse.ArgumentParser()

def Blur(img):
    return img.filter(ImageFilter.BLUR)
 
def GaussianBlur(img):
    return img.filter(ImageFilter.GaussianBlur(radius = random.randint(1, 6)))

def BoxBlur(img):
    return img.filter(ImageFilter.BoxBlur(radius = random.randint(1, 6)))

def Contrast(img):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(random.randint(0, 3))

def pixelate(img):
    imgSmall = img.resize((256, 256))
    return imgSmall.resize(img.size,Image.NEAREST)

def rotate(img):
    return img.rotate(random.randint(1, 45))

def prespective(img):
    width, height = img.size
    m = -0.5
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    return img.transform((new_width, height), Image.AFFINE,
            (1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)

def translate(img):
    a = 1
    b = 0
    c = 0 
    d = 0
    e = 1
    f = 0 
    return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col = image.size
        ch = 3
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(col,row,ch)
        noisy = image + gauss
        return Image.fromarray(noisy)
    elif noise_typ == "s&p":
        row,col = image.size
        ch = 3
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.size]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.size]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.size
        ch = 3
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def Vgrid(img):
    W, H = img.size
    max_width=4
    mag=-1
    if mag<0 or mag>max_width:
            line_width = np.random.randint(1, max_width)
            image_stripe = np.random.randint(1, max_width)
    else:
        line_width = 1
        image_stripe = 3 - mag

    n_lines = H // (line_width + image_stripe) + 1
    draw = ImageDraw.Draw(img)
    for i in range(1, n_lines):
        y = image_stripe*i + line_width*(i-1)
        draw.line([(0,y), (W, y)], width=line_width, fill='black')
    return img

def Hgrid(img):
    W, H = img.size
    max_width=4
    mag=-1
    if mag<0 or mag>max_width:
        line_width = np.random.randint(1, max_width)
        image_stripe = np.random.randint(1, max_width)
    else:
        line_width = 1
        image_stripe = 3 - mag

    n_lines = W // (line_width + image_stripe) + 1
    draw = ImageDraw.Draw(img)
    for i in range(1, n_lines):
        x = image_stripe*i + line_width*(i-1)
        draw.line([(x,0), (x,H)], width=line_width, fill='black')
    return img

if __name__ == '__main__':
    parser.add_argument("-f", "--file", help = "Give a File Directory to Augment")
    parser.add_argument("-o", "--output", help = "Give an Output Directory")
    args = parser.parse_args()
    if(args.file == None):
        print('Give a File Directory')
    else:
        if(args.output == None):
            print('Give a Output Directory')
        else:
            isExist = os.path.exists(args.output)
            if(isExist == False):
                os.mkdir(args.output)
            for filename in os.listdir(args.file):
                file = filename.split('.')[0]
                img = Image.open(args.file+'/'+filename).convert(mode="RGB")
                Blur(img).save(args.output+ '/' + file + '_' + 'BLUR' + '.png')
                GaussianBlur(img).save(args.output+ '/' + file + '_' + 'GaussianBlur' + '.png')
                BoxBlur(img).save(args.output+ '/' + file + '_' + 'BoxBlur' + '.png')
                pixelate(img).save(args.output+ '/' + file + '_' + 'pixelate' + '.png')
                rotate(img).save(args.output+ '/' + file + '_' + 'rotate' + '.png')
                prespective(img).save(args.output+ '/' + file + '_' + 'prespective' + '.png')
                translate(img).save(args.output+ '/' + file + '_' + 'translate' + '.png')
                Vgrid(img).save(args.output+ '/' + file + '_' + 'Vgrid' + '.png')
                Hgrid(img).save(args.output+ '/' + file + '_' + 'Hgrid' + '.png')

        
        