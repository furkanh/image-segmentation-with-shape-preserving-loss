import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from src.operations import *

class ModelDetailsLogger:
    def __init__(self, path, model_name):
        self.path = path
        self.model_name = model_name
        file = open(os.path.join(self.path, 'models', '{}.log'.format(self.model_name)), 'w+')
        file.close()
        
    def log(self, ls):
        file = open(os.path.join(self.path, 'models', '{}.log'.format(self.model_name)), 'a')
        for string, value in ls:
            file.write('{}: {}\n'.format(str(string), str(value)))
        file.close()
        
class ModelResultsLogger:
    def __init__(self, path, model_names, data_type, models, generator):
        self.data_type = data_type
        self.model_names = model_names
        self.path = path
        self.models = models
        self.generator = generator
        self.generator.generator.batch_size = 1
        if not os.path.exists(os.path.join(self.path, 'output', self.data_type)):
            os.mkdir(os.path.join(self.path, 'output', self.data_type))
            
    def log(self):
        cmap = ListedColormap(['white', 'red', 'green', 'blue', 'cyan'])
        outputs = []
        for model in self.models:
            outputs.append(predict(model, self.generator)[0])
        for i, key in enumerate(self.generator.generator.global_data):
            num_of_rows = len(self.models)
            output = self.models[0].output
            output = [output] if not isinstance(output, list) else output
            num_of_cols = len(output) + 1
            plt.clf()
            fig, ax = plt.subplots(num_of_rows, num_of_cols, figsize=(5*num_of_cols, 5*num_of_rows))
            if num_of_rows==1:
                ax[0].imshow(np.argmax(self.generator.generator.global_data[key]['y_mask'], axis=-1), cmap=cmap), ax[0].set_title('y_true')
                ax[0].set_ylabel(self.model_names[0])
                for c in range(1, num_of_cols):
                    ax[c].imshow(np.argmax(outputs[0][c-1][i], axis=-1), cmap=cmap), ax[c].set_title(self.models[0].output[c-1].name)
            else:
                for r in range(num_of_rows):
                    ax[r,0].imshow(np.argmax(self.generator.generator.global_data[key]['y_mask'], axis=-1), cmap=cmap), ax[r,0].set_title('y_true')
                    ax[r,0].set_ylabel(self.model_names[r])
                    for c in range(1, num_of_cols):
                        ax[r,c].imshow(np.argmax(outputs[r][c-1][i], axis=-1), cmap=cmap), ax[r,c].set_title(self.models[r].output[c-1].name)
            plt.savefig(os.path.join(self.path, 'output', self.data_type, '{}_{}.png'.format(self.model_names, key).replace('/','_')), dpi=200)