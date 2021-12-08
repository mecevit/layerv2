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

    def add_entity(self, entity):
        if not hasattr(entity, '_type'):
            raise Exception(f"Function {entity} is not decoratored!")
        elif entity._type == "dataset":
            ent = Dataset(entity)
            self.entities[entity._name] = ent
            return ent
        elif entity._type == "model":
            ent = Model(entity)
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
        for entity in self.entities.values():
            if entity.name == name:
                return entity
        raise Exception(f"Entity '{name}' not found!")


class Model:
    result = None

    def __init__(self, func):
        if func:
            self.name = func._name
            self.pickled_func = cloudpickle.dumps(func)
            self.func = func

    def run(self):
        new_func = pickle.loads(self.pickled_func)
        self.result = new_func()

    def get_train(self):
        return self.get_entity()

    def get_entity(self):
        if self.name in datasets:
            return models[self.name]
        else:
            raise Exception(f"Entity {self.name} is not built!")


class Dataset:
    result = None

    def __init__(self, func):
        if func:
            self.name = func._name
            self.pickled_func = cloudpickle.dumps(func)
            self.func = func

    def run(self):
        new_func = pickle.loads(self.pickled_func)
        result = new_func()
        datasets[self.name] = result

    def to_pandas(self):
        return self.get_entity()

    def get_entity(self):
        if self.name in datasets:
            return datasets[self.name]
        else:
            raise Exception(f"Entity {self.name} is not built!")


def dataset(name):
    def inner(func):
        def wrapped(*args, **kwargs):
            Layer.entity_context = name
            print(f'* Building {Layer.entity_context}...')
            res = func(*args, **kwargs)
            # TODO save returning entity to catalog
            return res

        wrapped._type = "dataset"
        wrapped._name = name

        return wrapped

    return inner


def model(name):
    def inner(func):
        def wrapped(*args, **kwargs):
            Layer.entity_context = name
            print(f'* Training {Layer.entity_context}...')
            res = func(*args, **kwargs)
            # TODO save returning entity to catalog
            return res

        wrapped._type = "model"
        wrapped._name = name

        return wrapped

    return inner
