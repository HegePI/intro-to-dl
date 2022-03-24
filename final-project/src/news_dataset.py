import pathlib
import typing
import xml
import torch


class newsDataset(torch.utils.data.Dataset):
    '''
    Custom dataset for news data.\n
    It counts and preprocesses news data.\n
    NewsDataset class can be passed to dataloader to make it easier to fetch the data.\n
    '''

    def __init__(self, path: str, codes: str):
        self.path = path
        code_and_meaning = {}

        with open(codes, "r") as f:
            for line in f.readlines():
                if (line[0] != ';'):
                    parts = line.split("\t")
                    code_and_meaning[parts[0].strip()] = parts[1].strip()

        # store .xml file paths into list for faster lookup
        files = list(pathlib.Path(self.path).glob("**/*.xml"))

        self.files = files
        self.number_of_files = len(files)
        self.codes = code_and_meaning

    def __len__(self):
        return self.number_of_files

    def __getitem__(self, idx: int) -> typing.Tuple[str, list[str]]:

        f = self.files[idx]

        tree = xml.etree.ElementTree.parse(f)

        text = "\n".join(list(map(lambda x: x.text, tree.findall("text/p"))))

        codes_in_xml = [item for sublist in list(map(lambda x: list(x.attrib.values()), tree.findall(
            "metadata/codes/code"))) for item in sublist]

        codes = [code for code in codes_in_xml if code in list(
            self.codes.keys())]

        return (text, codes)
