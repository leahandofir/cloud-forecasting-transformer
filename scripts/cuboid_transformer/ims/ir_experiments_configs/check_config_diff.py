# Help script - helped us to make sure all configs were updated.

import yaml
from deepdiff import DeepDiff
import os


def yaml_as_dict(my_file):
    my_dict = {}
    with open(my_file, 'r') as fp:
        docs = yaml.safe_load_all(fp)
        for doc in docs:
            for key, value in doc.items():
                my_dict[key] = value
    return my_dict


if __name__ == '__main__':
    new_standard = yaml_as_dict("../cfg_ims.yaml")

    for filename in os.listdir(os.path.dirname(__file__)):
        if filename.endswith('.yaml'):
            yaml_dict = yaml_as_dict(filename)
            ddiff = DeepDiff(new_standard, yaml_dict, ignore_order=True)

            print("*" * 15 + f"comparing cfg_ims.yaml to {filename}" + "*" * 15 + "\n")

            if "dictionary_item_removed" in ddiff.keys():
                print("some keys are missing from one of the files:")
                print(ddiff["dictionary_item_removed"])
                print()

            if "dictionary_item_added" in ddiff.keys():
                print("some keys are missing from one of the files:")
                print(ddiff["dictionary_item_added"])
                print()

            print("whole diff result (including diffs in values!):")
            print(ddiff)
            print("\n")
