import os

def check_and_make_dir(path):
    path = os.path.abspath(path)
    print('checking... ', path)
    dirs = path.split('/')
    path = '/'
    for _dir in dirs:
        path = os.path.join(path, _dir)
        print('check', path, os.path.isdir(path))
        if not os.path.isdir(path):
            print('mkdir', path)
            os.mkdir(path)