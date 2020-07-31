import os
from alisuretool.Tools import Tools


def get_file(mask_path, result_path):
    result_files = []
    for mask_name in os.listdir(mask_path):
        a = os.path.join(mask_path, mask_name)
        b = os.path.join(result_path, mask_name)
        if os.path.exists(a) and os.path.exists(b):
            result_files.append("{} {}".format(b, a))
        pass
    return result_files


"""
./salmetric salmetric.txt 8
"""


if __name__ == '__main__':
    _result_files = get_file("./data/DUTS/DUTS-TE/DUTS-TE-Mask", "./results/test/run-7/t")
    _txt = "\n".join(_result_files)
    Tools.write_to_txt("salmetric.txt", _txt, reset=True)
    pass
