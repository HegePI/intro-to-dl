from typing import Tuple
from pathlib import Path
import xml.etree.ElementTree as ET
import math
import csv


class XmlToCsvWriter:
    """
    Class that handles the transformation of data from .xml to .csv files.
    """

    def __init__(self, data_dir: str, codes: str):
        self.data_dir = data_dir

        code_and_idx = {}

        with open(codes, "r") as f:
            for idx, line in enumerate(f.readlines()):
                if line[0] != ";":
                    parts = line.split("\t")
                    code_and_idx[parts[0].strip()] = idx
        self.codes = code_and_idx

    def get_codes(self) -> list[str]:
        return list(self.codes.keys())

    def get_idx(self, code: str) -> int:
        return self.codes[code]

    def write_data_to_csv_files(
        self, csv_sizes: Tuple[float, float, float]
    ) -> Tuple[str, str, str]:
        """
        Function which takes relative sizes of train, dev and test data sizes as input,
        writes data into .csv files
        and returns file paths to written .csv files
        """
        xml_files = list(Path(self.data_dir).glob("**/*.xml"))

        train_amount = math.floor(len(xml_files) * csv_sizes[0])
        dev_amount = math.floor(len(xml_files) * csv_sizes[1])
        test_amount = math.floor(len(xml_files) * csv_sizes[2])

        train_files = xml_files[:train_amount]
        dev_files = xml_files[-dev_amount:]
        test_files = xml_files[train_amount + 1 : -dev_amount]

        for file in train_files:
            self.write_to_csv(f"{self.data_dir}/train.csv", file, self.codes)

        for file in dev_files:
            self.write_to_csv(f"{self.data_dir}/dev.csv", file, self.codes)

        for file in test_files:
            self.write_to_csv(f"{self.data_dir}/test.csv", file, self.codes)

        return (
            f"{self.data_dir}/train.csv",
            f"{self.data_dir}/dev.csv",
            f"{self.data_dir}/test.csv",
        )

    def write_to_csv(self, csv_file_path: str, xml_file_path: str, codes):
        """
        Function, which reads the contents of the .xml file,
        cleans out unneeded whitespaces and
        writes the contents of the xml files text blocks and code blocks into .csv file
        """
        tree = ET.parse(xml_file_path)
        text = (
            " ".join(list(map(lambda x: x.text, tree.findall("text/p"))))
            .replace("\n", " ")
            .strip()
        )

        codes_in_xml = [
            item
            for sublist in list(
                map(
                    lambda x: list(x.attrib.values()),
                    tree.findall("metadata/codes/code"),
                )
            )
            for item in sublist
        ]

        idx = list(
            map(
                str,
                map(
                    self.get_idx,
                    [code for code in codes_in_xml if code in list(codes.keys())],
                ),
            )
        )
        # check if no labels for sample, and set 127 as NONE label
        if len(idx) == 0:
            idx = ["127"]

        res = " ".join(idx)

        with open(csv_file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([res, text])
