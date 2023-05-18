import os
import shutil
import re
from time import gmtime, strftime

def get_all_files(root, extension_list=['.csv'], sorted=False):

    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    if sorted:
        all_files.sort()
    return all_files


def copy_project_files(code_dir, save_path):

    print('copying files from {}'.format(code_dir))
    print('copying files to {}'.format(save_path))
    py_files = get_all_files(code_dir, extension_list=['.py', '.yaml', '.sh'])
    os.makedirs(save_path, exist_ok=True)
    for py_file in py_files:
        os.makedirs(os.path.dirname(py_file.replace(code_dir, save_path)), exist_ok=True)
        shutil.copy(py_file, py_file.replace(code_dir, save_path))

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_project_root(project_name='dcface'):
    root = os.getcwd().rsplit(project_name, 1)[0] + project_name
    return root

def get_task(project_name='dcface'):
    task = os.getcwd().split(project_name)[1].split('/')[-1]
    return task

def make_runname(prefix):
    current_time = strftime("%m-%d_0", gmtime())
    return f'{prefix}_{current_time}'

def make_output_dir(exp_root, task, runname):
    output_dir = os.path.join(exp_root, task, runname)
    is_taken = os.path.isdir(output_dir) and os.path.isfile(os.path.join(output_dir, 'src','train.py'))
    if is_taken:
        while True:
            cur_exp_number = int(output_dir[-2:].replace('_', ""))
            output_dir = output_dir[:-2] + "_{}".format(cur_exp_number+1)
            is_taken = os.path.isdir(output_dir) and os.path.isfile(os.path.join(output_dir, 'src', 'train.py'))
            if not is_taken:
                break
    return output_dir
