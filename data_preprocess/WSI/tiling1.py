import time, os
import numpy as np
from tqdm import tqdm
from statistics import mean, median
import os
import numpy as np
from pathlib import Path
# from numpy.lib.shape_base import tile
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from multiprocessing import Process, JoinableQueue
join = os.path.join()


def tiling(slide, slide_path, save_path,tile_size,overlap,ext):

    slide = open_slide(slide_path+slide+'.mrxs')
    print(slide.level_count)
    print(slide.level_dimensions)
    print(slide.properties.get('openslide.mpp-x'))


    '''
    tiles = DeepZoomGenerator(
        slide,
        tile_size=256,
        overlap=8,
        limit_bounds=False
    )
    print()

    max_level = tiles.level_count - 1
    level = max_level - int(np.log(4) / np.log(2))
    cols, rows = tiles.level_tiles[level]
    print(cols)
    print(rows)
    '''
    level = 0
    cols = 0
    rows = 0
    return level, cols, rows


# raw_data_path = '/vast/AI_team/youngjin/data/stomach/slide/stomach_slide_original/train/'
raw_data_path = '/vast/AI_team/sukmin/0424_breast_prostate/'

save_path = './saved_patch' 
ext = 'png'
tile_size = 2048
levells = []
cls = []
rls = []

BP_list = sorted(next(os.walk(raw_data_path))[1])

for organ in BP_list:
    print(organ)
    data_path = join(raw_data_path, organ)
    list = os.listdir(data_path)
    overlap = 128
    for slide in  list:

            level, cols, rows = tiling(slide, data_path, save_path, tile_size, overlap, ext)
            levells.append(level)
            cls.append(cols)
            rls.append(rows)




print(max(levells))
print(min(levells))
print(mean(levells))

print(max(cls))
print(min(cls))
print(mean(cls))

print(max(rls))
print(min(rls))
print(mean(rls))