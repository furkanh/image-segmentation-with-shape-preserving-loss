import matplotlib.pyplot as plt
import os
from keras.callbacks import *
from src.callbacks import *
from src.data_generators import *
from keras import backend as K
import GPUtil
import time
import pickle
from tqdm import tqdm

def allocate_GPU(gpu_num):
    device_list = GPUtil.getAvailable(limit=gpu_num)
    while len(device_list)!=gpu_num:
        print('Cannot allocate GPU. Waiting...')
        time.sleep(60)
        device_list = GPUtil.getAvailable(limit=gpu_num)
    devices = ''
    for device in device_list:
        devices += str(device) + ','
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    
def predict_generator(model, generator):
    if not isinstance(generator, LambdaGenerator):
        assert isinstance(generator, SegmentationDataset)
        generator = LambdaGenerator(generator, [])
    true_dict = {}
    img_dict = {}
    pred_list = []
    model_output = model.output
    if not isinstance(model_output, list):
        model_output = [model_output]
    for i in range(len(model_output)):
        pred_list.append({})
    num_dict = {}
    for x, y in generator:
        y_pred_list = model.predict(x)
        if not isinstance(y_pred_list, list):
            y_pred_list = [y_pred_list]
        for i in range(len(y['stride_stats'])):
            path, h, w = y['stride_stats'][i]
            if path not in true_dict:
                true_dict[path] = generator.generator.global_data[path]['y']
                img_dict[path] = generator.generator.global_data[path]['x']
                for j in range(len(y_pred_list)):
                    pred_list[j][path] = np.zeros(generator.generator.global_data[path]['y_mask'].shape[:-1]+(y_pred_list[j].shape[-1],))
                num_dict[path] = np.zeros(generator.generator.global_data[path]['y_mask'].shape[:-1]+(1,))
            for j in range(len(y_pred_list)):
                patch = pred_list[j][path][h:h+generator.generator.model_shape[0],w:w+generator.generator.model_shape[1]]
                pred_list[j][path][h:h+generator.generator.model_shape[0],w:w+generator.generator.model_shape[1]] += y_pred_list[j][i,:patch.shape[0],:patch.shape[1]]
            num_dict[path][h:h+generator.generator.model_shape[0],w:w+generator.generator.model_shape[1],:] += 1
    y_pred = []
    for j in range(len(pred_list)):
        y_pred.append([])
    y_true = []
    img = []
    for path in true_dict:
        for j in range(len(pred_list)):
            y_pred[j].append(pred_list[j][path]/num_dict[path])
        y_true.append(true_dict[path])
        img.append(img_dict[path])
    return img, y_pred, y_true
    
def predict(model, generators):
    if not isinstance(generators, list):
        generators = [generators]
    img, y_pred, y_true = predict_generator(model, generators[0])
    y_pred_all = y_pred
    y_true_all = y_true
    img_all = img
    for i in range(1, len(generators)):
        img, y_pred, y_true = predict_generator(model, generators[i])
        for j in range(len(y_true_all)):
            y_pred_all[j] += y_pred[j]
        y_true_all += y_true
        img_all += img
    return img_all, y_pred_all, y_true_all

def train(
    model,
    train_generator,
    validation_generator,
    epochs,
    callbacks,
):
    if not isinstance(train_generator, LambdaGenerator):
        assert isinstance(train_generator, SegmentationDataset)
        train_generator = LambdaGenerator(train_generator, [])
    if not isinstance(validation_generator, LambdaGenerator):
        assert isinstance(validation_generator, SegmentationDataset)
        validation_generator = LambdaGenerator(validation_generator, [])
    if train_generator.use_default_train:
        model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            callbacks=callbacks,
            epochs=epochs,
            verbose=1,
            use_multiprocessing=True,
            workers=10,
            shuffle=True
        )
    else:
        model.stop_training = False
        stateful_metrics = []
        for metric_name in model.metrics_names:
            stateful_metrics.append('val_{}'.format(metric_name))
        progbar = ProgbarLogger(stateful_metrics=stateful_metrics)
        baselogger = BaseLogger(stateful_metrics=stateful_metrics)
        params = {
            'verbose' : 1,
            'epochs' : epochs,
            'steps' : len(train_generator),
            'samples' : len(train_generator.generator.data),
            'metrics' : stateful_metrics + model.metrics_names
        }
        progbar.params = params
        baselogger.params = params
        callbacks = [baselogger] + callbacks + [progbar]
        for callback in callbacks:
            callback.set_model(model)
            callback.on_train_begin()
        for epoch in range(epochs):
            logs = {}
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            for batch, (x, y) in enumerate(train_generator):
                logs['size'] = x[list(x.keys())[0]].shape[0]
                for callback in callbacks:
                    callback.on_batch_begin(batch)
                metrics = model.train_on_batch(x, y)
                for metric_name, metric in zip(model.metrics_names, metrics):
                    logs[metric_name] = metric
                for callback in callbacks:
                    callback.on_batch_end(batch, logs=logs)
            metrics = model.evaluate_generator(validation_generator)
            for metric_name, metric in zip(model.metrics_names, metrics):
                logs['val_{}'.format(metric_name)] = metric
            train_generator.on_epoch_end()
            validation_generator.on_epoch_end()
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)
            if model.stop_training==True:
                break
        for callback in callbacks:
            callback.on_train_end({})
