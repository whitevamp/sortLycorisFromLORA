#!/usr/bin/python

# original script done by m99 from https://civitai.com/user/m99
# modified by whitevamp to allow the script to print
# the output to a file. ( its a pretty crud implementation but it works. or I think so.:) )

import json
import os
import sys
import glob
import shutil


def read_safetensor_header(file_name):
    try:
        with open(file_name, 'rb') as file:
            # The first 8 bytes of a safetensor file specify the json header length
            buf = file.read(8)
            header_size = int.from_bytes(buf, byteorder='little')
            if header_size > 10000000:
                print("header too big")
                return None
            buf = file.read(header_size)
        data = json.loads(buf)
        return data
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        return None
    except json.JSONDecodeError as e:
        print("error decoding json metadata")
        return None


def is_lyco(header):
    # First check the metadata
    lora_meta_tag_found = False
    metadata = header.get("__metadata__")
    if metadata is not None:
        ss_network_module = metadata.get("ss_network_module")
        if ss_network_module is not None:
            print("ss_network_module: " + ss_network_module)
            lora_meta_tag_found = "lora" in ss_network_module
    # Could not determine based on metadata. Check the model
    for k, v in header.items():
        if any(a in k for a in ['downsamplers', 'conv1', 'conv2']):
            if lora_meta_tag_found:
                print("Metadata indicates this is a lora but it looks like a LyCORIS")
            print("Classifying network as LyCORIS due to the following block: " + k)
            return True
    print("No LyCORIS blocks found")
    return False


def prepare_model_move(src_model_path, target_dir_path, src_root_dir):
    files_to_move = []
    model_name_with_ext = os.path.basename(src_model_path)
    model_name = model_name_with_ext[:-len(".safetensors")]
    src_dir = src_model_path[:-len(model_name_with_ext)]
    relative_dir = src_dir[len(src_root_dir) + 1:]
    for file in glob.glob(src_dir + model_name + ".*"):
        file_name = os.path.basename(file)
        dst = os.path.join(target_dir_path, relative_dir, file_name)
        files_to_move.append({"src": file, "dst": dst})
    return files_to_move


def main():
    if len(sys.argv) < 2:
        print("\nUsage: \n    python find_lyco.py [path_to_directory_to_scan]\nor\n")
        print("    python find_lyco.py [path_to_directory_to_scan] [target_lycoris_directory]\n")
        print("If a target directory is supplied, confirmation will be asked before moving files.")
        print("If no target directory is supplied, this script will not move any files.\n")
        return
    scan_dir = os.path.abspath(sys.argv[1])
    # scan_dir = os.path.abspath('G:\\AI_Generation\\stable-diffusion-webui-master\\models\\Lora\\')
    print("Scan directory: " + scan_dir)
    target_dir = None
    if len(sys.argv) > 2:
        target_dir = os.path.abspath(sys.argv[2])
        print("Target LyCORIS directory: " + target_dir)
    safetensor_files = []
    for root, dirs, files in os.walk(scan_dir):
        for file in files:
            if file.endswith(".safetensors"):
                safetensor_files.append(os.path.join(root, file))
    lyco_files = []
    lora_files = []
    try:

        for path in safetensor_files:
            print(path)

            # print(json.dumps(header, indent=4))
            # print(json.dumps(header['__metadata__'], indent=4))

            # modified by whitevamp
            # Open the file for writing.
            file = open('safetensor_files.txt', "w", encoding="utf-8")
            # reformat the entire list into a different format..
            # from this ['drive:\path\to\file1', 'drive:\path\to\file2', 'drive:\path\to\file3']
            # to this
            # drive:\path\to\file1
            # drive:\path\to\file2
            # drive:\path\to\file3
            content = '\n'.join(safetensor_files)
            # content = str(files)
            # write out the header to the file.
            file.write("----------------------------------\n")
            file.write("| Looking for .safetensors files |\n")
            file.write("----------------------------------\n")
            # write out the formatted list to a file.
            file.write(content)
            # close the file.
            file.close()

            header = read_safetensor_header(path)
            # continue going through the file list even if there is no header.
            if header is None:
                continue

            if is_lyco(header):
                lyco_files.append(path)
            else:
                lora_files.append(path)

    finally:
        for path in lora_files:
            print(path)

            # modified by whitevamp
            # Open the file for writing.
            file = open('lora_files.txt', "w", encoding="utf-8")
            # reformat the entire list into a different format..
            # from this ['drive:\path\to\file1', 'drive:\path\to\file2', 'drive:\path\to\file3']
            # to this
            # drive:\path\to\file1
            # drive:\path\to\file2
            # drive:\path\to\file3
            content = '\n'.join(lora_files)
            # write out the header to the file.
            file.write("----------------------------------\n")
            file.write("| Found the following LORA files |\n")
            file.write("----------------------------------\n")
            # write out the formatted list to a file.
            file.write(content)
            # close the file.
            file.close()

    for path in lyco_files:
        print(path)

        # modified by whitevamp
        # Open the file for writing.
        file = open('lyco_files.txt', "w", encoding="utf-8")
        # reformat the entire list into a different format..
        # from this ['drive:\path\to\file1', 'drive:\path\to\file2', 'drive:\path\to\file3']
        # to this
        # drive:\path\to\file1
        # drive:\path\to\file2
        # drive:\path\to\file3
        content = '\n'.join(lyco_files)
        # write out the header to the file.
        file.write("-------------------------------------\n")
        file.write("|  Found the following LyCORIS files |\n")
        file.write("--------------------------------------\n")
        # write out the formatted list to a file.
        file.write(content)
        # close the file.
        file.close()

    if target_dir is not None:
        print("---------------------------")
        print("| Preparing to move files |")
        print("---------------------------")
        files_to_move = []
        for path in lyco_files:
            files_to_move += prepare_model_move(path, target_dir, scan_dir)
        print("About to move the following files:")
        for m in files_to_move:
            print(m["src"] + " -> " + m["dst"])
        if input("Continue? y/[n]?") == "y":
            print("Moving files")
            for m in files_to_move:
                print("Moving " + m["src"] + " to " + m["dst"])
                try:
                    dir_path = os.path.dirname(m["dst"])
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    shutil.move(m["src"], m["dst"])
                except KeyboardInterrupt:
                    print("Canceling")
                    raise
                except Exception:
                    print("Failed to move file")
                    raise
            print("Done!")
        else:
            print("Move aborted")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
