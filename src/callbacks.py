from keras.callbacks import *
import numpy as np
import os

class AuxModelCheckpoint(Callback):
    def __init__(self, filepath):
        super(AuxModelCheckpoint, self).__init__()
        self.filepath = filepath
        
        path, filename = os.path.split(self.filepath)
        new_filename = filename.replace('-temp','')
        new_filepath = os.path.join(path, new_filename)
        
        self.new_filepath = new_filepath
    
    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists(self.filepath):
            os.rename(self.filepath, self.new_filepath)
