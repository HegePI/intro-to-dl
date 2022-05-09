import json


class Parameters:
    """Class to load and access params in hyperparameters.json, defaults to "base" mode"""

    def __init__(self, mode="base"):
        self.mode = mode

        with open(
            "/home/heikki/koulu/intro-to-dl/final-project/src/hyperparameters.json"
        ) as file:
            self.params = json.loads(file.read())

    def get(self, attr):
        """Get a named attribute"""
        return self.params[self.mode][attr]

    def set_mode(self, mode):
        """Set mode defined in hyperparameters.json"""
        self.mode = mode

    def get_mode(self):
        """Get mode"""
        return self.mode
