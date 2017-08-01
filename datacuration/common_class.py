class MetaDatasetClass:
    def __init__(self, filepath, name, country=None, city=None, actor=None,
                 exogfilepath=None):
        self.filepath = filepath
        self.name = name
        self.country = country
        self.city = city
        self.actor = actor
        self.exog_filepath = exogfilepath


class MetaModelClass:
    def __init__(self, name, param):
        self.name = name
        self.param = param
