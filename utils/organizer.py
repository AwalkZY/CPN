from utils.accessor import load_json


class ConfigOrganizer(object):
    def __init__(self):
        super(ConfigOrganizer, self).__init__()
        self.config = {}

    def load_config(self, path, name):
        config = load_json("./config/" + path)
        self.config.update({name: config})

    def flatten_config(self, path, prefix, common_name):
        config = load_json("./config/" + path)
        assert common_name in config, "Invalid common config name!"
        common_config = config[common_name]
        for configName in config:
            if configName != common_name:
                config[configName].update(common_config)
                self.config.update({prefix + "_" + configName: config[configName]})

    def fetch_config(self, name):
        assert name in self.config, "Invalid config name!"
        return self.config[name]

    def concat_config(self, target_name, source_name):
        assert target_name in self.config, "Invalid target config name!"
        assert source_name in self.config, "Invalid source config name!"
        self.config[target_name].update(self.config[source_name])


class ModelOrganizer(object):
    def __init__(self):
        super(ModelOrganizer, self).__init__()
        self.model = {}
        self.data = {}

    def save_model(self, model, name, data, criterion, greater=True):
        assert criterion in data, "Incompatible criterion name!"
        if (name not in self.model) or ((self.data[name][criterion] > data[criterion]) == greater):
            self.model.update({name: model})
            self.data.update({name: data})
        return self.data[name]

    def fetch_model(self, name):
        assert name in self.model, "Invalid model name!"
        return self.model[name], self.data[name]


configOrganizer = ConfigOrganizer()
modelOrganizer = ModelOrganizer()
