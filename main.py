import config
import os
from tqdm import tqdm

def main(ScaleType, Type):
    if config.dataprocess:
        from dataprocess import data_reader
        # build_graph.save_net()
        data_reader.load_dataset()
        # data_reader

    exit()


    os.makedirs(config.save_path)
    bar = tqdm(range(config.gs), total=config.gs)

    for step in bar:
        pass


if __name__ == '__main__':
    main('min', 'train')