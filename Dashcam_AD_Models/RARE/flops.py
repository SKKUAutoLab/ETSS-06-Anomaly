from ultralytics import YOLOv10

''' YOLOv10 Model Attributes and Methods
Attributes:
    callbacks (dict): A dictionary of callback functions for various events during model operations.
    predictor (BasePredictor): The predictor object used for making predictions.
    model (nn.Module): The underlying PyTorch model.
    trainer (BaseTrainer): The trainer object used for training the model.
    ckpt (dict): The checkpoint data if the model is loaded from a *.pt file.
    cfg (str): The configuration of the model if loaded from a *.yaml file.
    ckpt_path (str): The path to the checkpoint file.
    overrides (dict): A dictionary of overrides for model configuration.
    metrics (dict): The latest training/validation metrics.
    session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
    task (str): The type of task the model is intended for.
    model_name (str): The name of the model.

Methods:
    __call__: Alias for the predict method, enabling the model instance to be callable.
    _new: Initializes a new model based on a configuration file.
    _load: Loads a model from a checkpoint file.
    _check_is_pytorch_model: Ensures that the model is a PyTorch model.
    reset_weights: Resets the model's weights to their initial state.
    load: Loads model weights from a specified file.
    save: Saves the current state of the model to a file.
    info: Logs or returns information about the model.
    fuse: Fuses Conv2d and BatchNorm2d layers for optimized inference.
    predict: Performs object detection predictions.
    track: Performs object tracking.
    val: Validates the model on a dataset.
    benchmark: Benchmarks the model on various export formats.
    export: Exports the model to different formats.
    train: Trains the model on a dataset.
    tune: Performs hyperparameter tuning.
    _apply: Applies a function to the model's tensors.
    add_callback: Adds a callback function for an event.
    clear_callback: Clears all callbacks for an event.
    reset_callbacks: Resets all callbacks to their default functions.
    _get_hub_session: Retrieves or creates an Ultralytics HUB session.
    is_triton_model: Checks if a model is a Triton Server model.
    is_hub_model: Checks if a model is an Ultralytics HUB model.
    _reset_ckpt_args: Resets checkpoint arguments when loading a PyTorch model.
    _smart_load: Loads the appropriate module based on the model task.
    task_map: Provides a mapping from model tasks to corresponding classes.
'''


model = YOLOv10('yolov10s.yaml')
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.fuse()

# print model details
print(model)
# print(model.info(detailed=True, verbose=True))