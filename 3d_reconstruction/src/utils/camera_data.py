import json

class CameraData:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = None

    def load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def analyse_data(self):
        if self.data is None: self.data = self.load_data()

        