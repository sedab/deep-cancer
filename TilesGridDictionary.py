import re 
import numpy as np
import os
import pickle

'''
Author: @edufierro

Capstone project

Purpose: Get dictionary with files:[2Darray tiles, type of cancer]
'''

### File params ###
file_dir_normal = "/beegfs/jmw784/Capstone/LungTilesSorted/Solid_Tissue_Normal"
file_dir_LUAD = "/beegfs/jmw784/Capstone/LungTilesSorted/TCGA-LUAD"
file_dir_LUSC = "/beegfs/jmw784/Capstone/LungTilesSorted/TCGA-LUSC"
out_dir = "/beegfs/jmw784/Capstone"

def getCoords(tile_list): 
    
    '''
    Given a list of tiles, with format: 
    [test, valid, train]_NAME_x_y.jpeg
    Returns a two list of same size with xcoords and y coords    
    '''
    
    xcoords = [re.split("_", i)[-2] for i in tile_list]
    xcoords = list(map(int, xcoords))
    ycoords = [re.split("_", i)[-1] for i in tile_list]
    ycoords = [re.sub(".jpeg", "", i) for i in ycoords]
    ycoords = list(map(int, ycoords))
    
    return xcoords, ycoords


def fileCleaner(tile_list): 
    
    '''
    Given a list of tiles, remove coords ("_X_Y_") and ".jpeg" termination
    '''
    
    tile_list = [re.sub("_[0-9]*_[0-9]*.jpeg", "", x) for x in tile_list]
    
    return (tile_list)

def get2Darray(xcoords, ycoords, tiles_input): 
    
    '''
    Given a list of xcoords, ycoords and files, returns a 2D array where each file
       correspond to the pair of coords
    '''
    
    xmax = max(xcoords) + 1
    ymax = max(ycoords) + 1
    tiles_output = np.empty((ymax, xmax), dtype=np.dtype((str, 100)))
    for i in range(0,len(xcoords)): 
        tiles_output[ycoords[i], xcoords[i]] = tiles_input[i]
        
    return tiles_output 

def main():
    
    print("Importing File Names...")
    tiles_normal = os.listdir(file_dir_normal)
    tiles_LUAD = os.listdir(file_dir_LUAD)
    tiles_LUSC = os.listdir(file_dir_LUSC)
    
    original_files_normal = fileCleaner(tiles_normal)
    original_files_LUAD = fileCleaner(tiles_LUAD)
    original_files_LUSC = fileCleaner(tiles_LUSC)
    
    #type_dict = {'TCGA-LUSC': 2, 'Solid_Tissue_Normal': 0, 'TCGA-LUAD': 1}

    main_dict = {}
    
    print("Updating dict for normal files ...")
    for file in set(original_files_normal):
        
        index_list = [i for i, x in enumerate(original_files_normal) if x==file]
        tiles = [tiles_normal[i] for i in index_list]
        xs, ys = getCoords(tiles)
        tiles_array = get2Darray(xs, ys, tiles)
        loop_dict = {file:[tiles_array, 0]}
        main_dict.update(loop_dict)
    
    print("Updating dict for LUAD files ...")
    for file in set(original_files_LUAD):
        
        index_list = [i for i, x in enumerate(original_files_LUAD) if x==file]
        tiles = [tiles_LUAD[i] for i in index_list]
        xs, ys = getCoords(tiles)
        tiles_array = get2Darray(xs, ys, tiles)
        loop_dict = {file:[tiles_array, 1]}
        main_dict.update(loop_dict) 
    
    print("Updating dict for LUSC files ...")
    for file in set(original_files_LUSC):
        
        index_list = [i for i, x in enumerate(original_files_LUSC) if x==file]
        tiles = [tiles_LUSC[i] for i in index_list]
        xs, ys = getCoords(tiles)
        tiles_array = get2Darray(xs, ys, tiles)
        loop_dict = {file:[tiles_array, 2]}
        main_dict.update(loop_dict)    
    
    pickle.dump(main_dict, open( out_dir + "Lung_FileMappingDict.p", "wb" ) ) 
    print("Dictionary Ready!!! Saved as pickle in: \n {}/Lung_FileMappingDict.p".format(out_dir))
    
    return main_dict


if __name__ == '__main__':
    main()
