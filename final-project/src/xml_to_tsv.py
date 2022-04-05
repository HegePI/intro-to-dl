from typing import Tuple
from pathlib import Path
import xml.etree.ElementTree as ET
import math


def write_to_tsv(tsv_file_path: str, xml_file_path: str, codes):
    """
    Function, which reads the contents of the .xml file,
    cleans out unneeded whitespaces and
    writes the contents of the xml files text blocks and code blocks into .tsv file
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

    filtered_codes = " ".join(
        [code for code in codes_in_xml if code in list(codes.keys())]
    )

    f = open(tsv_file_path, "a")
    f.write(f'"{text}"\t"{filtered_codes}"\n')
    f.close()


class XmlToTsv:
    """
    Class that handles the transformation of data from .xml to .tsv files.
    """

    def __init__(self, data_dir: str, codes: str):
        self.data_dir = data_dir

        code_and_meaning = {}

        with open(codes, "r") as f:
            for line in f.readlines():
                if line[0] != ";":
                    parts = line.split("\t")
                    code_and_meaning[parts[0].strip()] = parts[1].strip()
        self.codes = code_and_meaning

    def write_data_to_tsv_files(
        self, tsv_sizes: Tuple[float, float, float]
    ) -> Tuple[str, str, str]:
        """
        Function which takes relative sizes of train, dev and test data sizes as input,
        writes data into .tsv files
        and returns file paths to written .tsv files
        """
        xml_files = list(Path(self.data_dir).glob("**/*.xml"))

        train_amount = math.floor(len(xml_files) * tsv_sizes[0])
        dev_amount = math.floor(len(xml_files) * tsv_sizes[1])
        test_amount = math.floor(len(xml_files) * tsv_sizes[2])

        train_files = xml_files[:train_amount]
        dev_files = xml_files[-dev_amount:]
        test_files = xml_files[train_amount + 1 : -dev_amount]

        for file in train_files:
            write_to_tsv(f"{self.data_dir}/train.tsv", file, self.codes)

        for file in dev_files:
            write_to_tsv(f"{self.data_dir}/dev.tsv", file, self.codes)

        for file in test_files:
            write_to_tsv(f"{self.data_dir}/test.tsv", file, self.codes)

        return (
            f"{self.data_dir}/train.tsv",
            f"{self.data_dir}/dev.tsv",
            f"{self.data_dir}/test.tsv",
        )
