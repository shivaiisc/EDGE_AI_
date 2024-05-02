import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from matplotlib import pyplot as plt
from model import basic_classifier as classifier 
import timeit
import os 

def main():
    MODEL_INPUT_WIDTH = 48 
    MODEL_INPUT_HEIGHT = 48

    data_dir = './data/train'

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    interpolation="bilinear",
    image_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    interpolation="bilinear",
    image_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
    )

    class_names = train_ds.class_names

    rescale = tf.keras.layers.Rescaling(1./255, offset= -1)
    train_ds = train_ds.map(lambda x, y: (rescale(x), y))
    val_ds   = val_ds.map(lambda x, y: (rescale(x), y))


    augmen = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    ])

    train_ds = train_ds.map(lambda x, y: (augmen(x), y))
    val_ds   = val_ds.map(lambda x, y: (augmen(x), y))


    lr = 0.0005
    classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])



    classifier.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
    )

    classifier.save('classifier')


    history = classifier.history.history
    fig, ax  = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(history['loss'], label='train')
    ax.plot(history['val_loss'], label='val')
    ax.legend()
    fig.savefig('./res/loss.png')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10,8))
    ax.plot(history['accuracy'], label='train')
    ax.plot(history['val_accuracy'], label='val')
    ax.legend()
    fig.savefig('./res/acc.png')
    plt.close()

    test_dir = "./data/validation"

    test_ds = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                        interpolation="bilinear",
                                                        image_size=(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
    test_ds  = test_ds.map(lambda x, y: (rescale(x), y))

    loss, accuracy = classifier.evaluate(test_ds)
    print("Accuracy", accuracy)

    for idx, (image, label) in enumerate(test_ds):
        for i in range(len(image)):
            break
            img = image[i].numpy()
            img = (img - img.min()) / (img.max() - img.min())
            predictions = classifier.predict(image)
            title = './res/predicts/' + str(idx) 
            title += f"Prediction: {class_names[np.argmax(predictions[i])]}"
            title += '.png'
            plt.imsave(title, img)

    converter = tf.lite.TFLiteConverter.from_saved_model('classifier')
    tflite_model = converter.convert()
    print(f"Size of the tflite model without quantization: {len(tflite_model)} bytes")

    repr_ds = val_ds.unbatch()

    def representative_data_gen():
        for i_value, o_value in repr_ds.batch(1).take(48):
            yield [i_value]

    converter = tf.lite.TFLiteConverter.from_saved_model('classifier')
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_data_gen)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    tfl_model = converter.convert()

    size_tfl_model = len(tfl_model)
    print(len(tfl_model), "bytes")

    interpreter = tf.lite.Interpreter(model_content=tfl_model)
    interpreter.allocate_tensors()

    i_details = interpreter.get_input_details()[0]
    o_details = interpreter.get_output_details()[0]

    i_quant = i_details["quantization_parameters"]
    i_scale      = i_quant['scales'][0]
    i_zero_point = i_quant['zero_points'][0]

    test_ds0 = test_ds.unbatch()

    num_correct_samples = 0
    num_total_samples   = len(list(test_ds0.batch(1)))
    total_time = 0

    for i_value, o_value in test_ds0.batch(1):
        i_value = (i_value / i_scale) + i_zero_point
    i_value = tf.cast(i_value, dtype=tf.int8)
    interpreter.set_tensor(i_details["index"], i_value)
    interpreter.invoke()
    
    start_time = timeit.default_timer()
    o_pred = interpreter.get_tensor(o_details["index"])[0]
    end_time = timeit.default_timer()
    inference_time = end_time - start_time
    total_time += inference_time

    if np.argmax(o_pred) == o_value:
        num_correct_samples += 1

    print("Accuracy:", num_correct_samples/num_total_samples)
    print("Average Inference Time:", total_time/num_total_samples, "seconds")

    open("model.tflite", "wb").write(tfl_model)

    os.system('xxd -c 60 -i model.tflite > emoji_classifier.h')

    test_ds0 = test_ds.unbatch()
    y_true = []
    y_pred = []

    for i_value, o_value in test_ds0.batch(1):
        i_value = (i_value / i_scale) + i_zero_point
        i_value = tf.cast(i_value, dtype=tf.int8)
        interpreter.set_tensor(i_details["index"], i_value)
        interpreter.invoke()
        o_pred = interpreter.get_tensor(o_details["index"])[0]
        y_true.append(o_value.numpy()[0])
        y_pred.append(np.argmax(o_pred))

    cm = confusion_matrix(y_true, y_pred)
    dummy = np.zeros((100, 100), dtype=np.uint8)
    dummy[:50, :50] = cm[0,0]
    dummy[50:, 50:] = cm[1,1]
    dummy[:50, 50:] = cm[0,1]
    dummy[50:, :50] = cm[1,0]
    plt.imsave('./res/confusion_img.png', dummy, cmap=plt.cm.Blues)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    fig.savefig('./res/confusion_mat.png')

    for i in range(3):
        print('\n')
    print('='*20, 'Successfull Run', '='*20)








if __name__ == '__main__': 

    main()
