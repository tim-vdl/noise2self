import numpy as np
import skimage
from pathlib import Path
from data_utils import get_files_in_folder, load_2D_nd2_img, load_custom_config

def nd2png(input_path, output_path, channel_names, file_extension_out='.png', size_out=None):
    if not output_path.is_dir():
        output_path.mkdir()
    files = get_files_in_folder(input_path, file_extension='.nd2')
    n_files = len(files)
    for i, file_name_in in enumerate(files):
        print(f'{i+1}/{n_files}) {file_name_in}')
        name = file_name_in.stem 
        images = load_2D_nd2_img(input_path.joinpath(file_name_in))
        
        for c, channel in enumerate(channel_names):
            img = images[c,:,:]
            if size_out is not None:
                img = skimage.transform.resize(img, size_out)
            output_path_channel = output_path.joinpath(f'{channel}')
            if not output_path_channel.is_dir():
                output_path_channel.mkdir()
            file_name_out =  output_path_channel.joinpath(name + f'{file_extension_out}')
            skimage.io.imsave(file_name_out, img)

def main():
    custom_config = load_custom_config()
    input_path = Path(custom_config['data_path']).joinpath('Images/NikonHTM_1321N1_vs_SHSY5Y/20211105_183546_943')
    output_path = Path(custom_config['data_path']).joinpath('Images/NikonHTM_1321N1_vs_SHSY5Y/tif_channels')
    channel_names = ['channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5']
    nd2png(input_path, output_path, channel_names, size_out=(256, 256))


if __name__ == '__main__':
    main()