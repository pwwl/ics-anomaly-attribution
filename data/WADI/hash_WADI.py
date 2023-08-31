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

NORMAL_PATH = 'WADI_14days.csv'
ATTACK_PATH = 'WADI_attackdata.csv'
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
expected_normal_hash = "2d317fc8bd26a0d3ad72aac4f23edb0b28dfab366c7ee2104f2d527e1f972893"
if normal_hash != expected_normal_hash:
    print(f"IMPORTANT: the hash of {NORMAL_PATH} has changed from the expected hash, meaning process_WADI.py will no longer function correctly.")
    raise SystemExit("Please submit an Issue at https://github.com/pwwl/ics-anomaly-detection/issues/ to make us aware that the WADI dataset has changed. Sorry for the inconvenience!")
attack_hash = calculate_hash(ATTACK_PATH)
expected_attack_hash = "1211973d5b86aca6cba01c1603ed620b306912e6db6ecac1ad441c7cb692282d"
if attack_hash != expected_attack_hash:
    print(f"IMPORTANT: the hash of {ATTACK_PATH} has changed from the expected hash, meaning process_WADI.py will no longer function correctly.")
    raise SystemExit("Please submit an Issue at https://github.com/pwwl/ics-anomaly-detection/issues/ to make us aware that the WADI dataset has changed. Sorry for the inconvenience!")

print("Hashes match, WADI dataset is unchanged. Proceed to using process_WADI.py")
