"""

   Copyright 2020 Lujo Bauer, Clement Fung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import hashlib
import csv

NORMAL_PATH = 'SWaT_Dataset_Normal_v1.csv'
ATTACK_PATH = 'SWaT_Dataset_Attack_v0.csv'
BLOCK_SIZE  = 256 * 128

def calculate_hash(file_path):

    hash = hashlib.sha256()
    try:
        with open(file_path, 'r') as file:

            csv_reader = csv.reader(file)

            for row in csv_reader:
                row_string = ''.join(row)
                if row_string == "":
                    row_bytes = b"EMPTY_ROW_SEPARATOR"
                else:
                    row_bytes = row_string.encode('utf-8')
                hash.update(row_bytes)
    except FileNotFoundError:
        raise SystemExit(f"Unable to find file {file_path}. Did you request the dataset from iTrust and download the file as a CSV correctly?")

    return hash.hexdigest()

print("Calculating hashes...")

normal_hash = calculate_hash(NORMAL_PATH)
expected_normal_hash = "3710e89098e099fc22ba25457a4c45428865fe94d9a868c937ed0feb232c8916"
if normal_hash != expected_normal_hash:
    print(f"IMPORTANT: the hash of {NORMAL_PATH} has changed from the expected hash, meaning process_SWaT.py will no longer function correctly.")
    raise SystemExit("Please submit an Issue at https://github.com/pwwl/ics-anomaly-detection/issues/ to make us aware that the SWaT dataset has changed. Sorry for the inconvenience!")
attack_hash = calculate_hash(ATTACK_PATH)
expected_attack_hash = "59556a9c0d03739e7644c1b669d5d16137e65b7b3df723a406956ec062c743a9"
if attack_hash != expected_attack_hash:
    print(f"IMPORTANT: the hash of {ATTACK_PATH} has changed from the expected hash, meaning process_SWaT.py will no longer function correctly.")
    raise SystemExit("Please submit an Issue at https://github.com/pwwl/ics-anomaly-detection/issues/ to make us aware that the SWaT dataset has changed. Sorry for the inconvenience!")

print("Hashes match, SWaT dataset is unchanged. Proceed to using process_SWaT.py")
