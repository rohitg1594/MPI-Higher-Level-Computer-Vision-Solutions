import sys
from showVisual import *
from vocab import *
from index import *
import numpy as np
from utils import *

if __name__ == "__main__":
    train = sys.argv[2]
    
    if sys.argv[1] == "show":
        random_path = list(np.random.randint(0, 8, size=4))
        Show_visual_word(random_path, train)

    if sys.argv[1] == "index":
        Invert_file_index(train)

    if sys.argv[1] == "retr":
        test = sys.argv[3]
        img_i = int(sys.argv[4])
        img = load_images(test)[0][img_i]
        retrieval(train, img)
