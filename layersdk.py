import pickle

import os
import cloudpickle

# Simple persistency for built entities
datasets = {}
models = {}


class Layer:
    entities = {}
    entity_context = None

    def __init__(self, project_name, environment=None):
        self.project_name = project_name
        self.environment = environment

    def setup(self):
        if not self.environment:
            return
        if os.path.exists(self.environment):
            file1 = open(self.environment, 'r')
            for lib in file1.readlines():
                print(f"Layer Infra: Installing {lib.strip()}...")
        else:
            print(f"Environment file not found: {self.environment}")

    def log_parameter(self, metric, value):
        print(f"\t{Layer.entity_context} > Parameter > {metric}:{value}")

    def log_metric(self, metric, value):
        print(f"\t{Layer.entity_context} > Metric >{metric}:{value}")

    def log(self, message):
        print(f"\t{Layer.entity_context} > {message}")

    def list_entities(self):
        for ds in self.entities:
            print('\t' + ds)

    def add_entity(self, entity):
        if not hasattr(entity, '_type'):
            raise Exception(f"Function {entity} is not decoratored!")
        elif entity._type == "dataset":
            ent = DatasetDefinition(entity)
            self.entities[entity._name] = ent
            return ent
        elif entity._type == "model":
            ent = ModelDefinition(entity)
            self.entities[entity._name] = ent
            return ent

    def run(self, entities):
        print(f"--- Layer Infra: Running Project: {self.project_name} ---")
        self.setup()

        if isinstance(entities, list):
            for entity in entities:
                ent = self.add_entity(entity)
                ent.run()
        else:
            ent = self.add_entity(entities)
            ent.run()

        print(f"--- Layer Infra: Run Complete! ---")

    def get_dataset(self, name):
        return self.get_entity(name)

    def get_model(self, name):
        return self.get_entity(name)

    def get_entity(self, name):
        for entity_name in self.entities:
            if entity_name == name:
                return self.entities[entity_name]
        raise Exception(f"Entity '{name}' not found!")


class ModelDefinition:

    def __init__(self, func):
        if func:
            self.name = func._name
            self.pickled_func = cloudpickle.dumps(func)
            self.func = func

    def run(self):
        print(f'* Training {self.name}...')
        new_func = pickle.loads(self.pickled_func)
        for dependency in new_func._depends:
            print("\t\tDependency: ", dependency)
        result = new_func()
        models[self.name] = result

    def get_train(self):
        return self.get_entity()

    def get_entity(self):
        if self.name in models:
            return models[self.name]
        else:
            raise Exception(f"Entity {self.name} is not built!")


class DatasetDefinition:
    def __init__(self, func):
        if func:
            self.name = func._name
            self.pickled_func = cloudpickle.dumps(func)
            self.func = func

    def run(self):
        new_func = pickle.loads(self.pickled_func)
        print(f'* Building {self.name}...')
        for dependency in new_func._depends:
            print("\t\tDependency: ", dependency)
        result = new_func()
        datasets[self.name] = result

    def to_pandas(self):
        return self.get_entity()

    def get_entity(self):
        if self.name in datasets:
            return datasets[self.name]
        else:
            raise Exception(f"Entity {self.name} is not built!")


## =========== DECORATORS ======================================================

def dataset(name, depends=[], fabric=None):
    def inner(func):
        def wrapped(*args, **kwargs):
            Layer.entity_context = name
            res = func(*args, **kwargs)
            # TODO save returning entity to catalog
            return res

        wrapped._type = "dataset"
        wrapped._name = name
        wrapped._depends = depends
        wrapped._transformation_type = type
        wrapped._fabric = fabric

        return wrapped

    return inner


def model(name, depends=[], fabric=None):
    def inner(func):
        def wrapped(*args, **kwargs):
            Layer.entity_context = name
            res = func(*args, **kwargs)
            # TODO save returning entity to catalog
            return res

        wrapped._type = "model"
        wrapped._name = name
        wrapped._depends = depends

        return wrapped

    return inner


## =========== DEPENDENCY DEFINITIONS ==========================================

class File:
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"File('{self.path}')"


class Directory:
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return f"Directory('{self.path}')"


class Dataset:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Dataset('{self.name}')"


class Model:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Model('{self.name}')"


## =========== SOURCE DEFINITIONS ==============================================


class Datasource:
    def __init__(self, integration_name, table_name):
        self.integration_name = integration_name
        self.table_name = table_name


class SQL:
    def __init__(self, query):
        self.query
