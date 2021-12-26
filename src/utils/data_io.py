import pickle
import h5py
import os

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def load_h5(file_path, transform_slash=True, parallel=False):
    """load the whole h5 file into memory (not memmaped)
    TODO: Loading data in parallel
    """
    with h5py.File(file_path, 'r') as f:
        # if parallel:
        #     Parallel()
        data = {k if not transform_slash else k.replace('+', '/'): v.__array__() \
                    for k, v in f.items()}
    return data

def save_h5(dict_to_save, filename, transform_slash=True):
    """Saves dictionary to hdf5 file"""
    with h5py.File(filename, 'w') as f:
        for key in dict_to_save:  # h5py doesn't allow '/' in object name (will leads to sub-group)
            f.create_dataset(key.replace('/', '+') if transform_slash else key,
                             data=dict_to_save[key])


def load_calib(calib_fullpath_list, subset_index=None):
    """Load all IMC calibration files and create a dictionary."""

    calib = {}
    if subset_index is None:
        for _calib_file in calib_fullpath_list:
            img_name = os.path.splitext(os.path.basename(_calib_file))[0].replace(
                "calibration_", ""
            )
            # _calib_file.split(
            #     '/')[-1].replace('calibration_', '')[:-3]
            # # Don't know why, but rstrip .h5 also strips
            # # more then necssary sometimes!
            # #
            # # img_name = _calib_file.split(
            # #     '/')[-1].replace('calibration_', '').rstrip('.h5')
            calib[img_name] = load_h5(_calib_file)
    else:
        for idx in subset_index:
            _calib_file = calib_fullpath_list[idx]
            img_name = os.path.splitext(os.path.basename(_calib_file))[0].replace(
                "calibration_", ""
            )
            calib[img_name] = load_h5(_calib_file)
    return calib