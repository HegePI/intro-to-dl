import json


class Parameters:
    def __init__(self, mode="base"):
        self.mode = mode

        with open("~/intro-to-dl/final-project/src/hyperparameters.json") as file:
            self.params = json.loads(file.read())

    def get(self, attr):
        return self.params[self.mode][attr]

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode
