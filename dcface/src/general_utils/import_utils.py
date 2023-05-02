import importlib

def load_task(task_name, file_name):
    module = importlib.import_module('tasks.' + task_name + '.' + file_name)
    return module
    
