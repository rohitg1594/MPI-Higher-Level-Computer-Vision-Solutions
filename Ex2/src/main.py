import sys
from showVisual import *
from vocab import *
from index import *
import numpy as np
from utils import *
from classification import *
from sklearn.svm import SVC

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

    if sys.argv[1] == "clas":
        train = sys.argv[2]
        test = sys.argv[3]
        vocab_size = sys.argv[4]
        
        codebook, _ = create_codebook(train, int(vocab_size))
        
        train_data, train_labels = extract_bow_features(train, codebook)
        test_data, test_labels = extract_bow_features(test, codebook)
        
        train_labels = [1 if label == 'airplane' else 0 for label in train_labels]        
        test_labels = [1 if label == 'airplane' else 0 for label in test_labels]        

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for kernel in kernels:
            svm = SVC(kernel=kernel)
            svm.fit(train_data, train_labels)
            test_predictions = svm.predict(test_data)
        
            print('{} : {}'.format(kernel,np.sum(test_predictions == test_labels)/len(test_predictions) * 100))
        
