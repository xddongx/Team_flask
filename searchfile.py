import os
def search(dirname):
    filenames = os.listdir(dirname)
    list = []
    for filename in filenames:
        full_filename = os.path.join(filename)
        list.append(full_filename)
    print(list)
    return list

search("./static/image/member")