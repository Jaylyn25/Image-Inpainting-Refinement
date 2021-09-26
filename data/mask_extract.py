import scipy as sp
from PIL import Image
import numpy as np

def extract_rgb(in_img,color_rgb=[255,255,255]):
    [rol, row, mut] = in_img.shape # 180 320 3
    out_img = np.zeros((rol, row))
    for x in range(rol):
        for y in range(row):
            R = in_img[x, y, 0]
            G = in_img[x, y, 1]
            B = in_img[x, y, 2]
            if ((R == color_rgb[0]) and (G == color_rgb[1]) and (B == color_rgb[2])):
                out_img[x, y] = 255
                if (x!=0) and (x!=rol-1) and (y!=0) and (y!=row-1):
                    for k1 in range(-1, 2):
                        for k2 in range(-1, 2):
                            out_img[x + k1, y + k2] = 255
    return out_img

#img = Image.open('G:/Applied Machine Learning/MSc project/Code/data_extreme/no.20/de/20.png').convert('RGB')
#img = np.array(img)
#out = extract_rgb(img)

#out = Image.fromarray(out.astype(np.uint8))
#out.save("G:/Applied Machine Learning/MSc project/Code/data_extreme/no.20/mask/geeks.png")