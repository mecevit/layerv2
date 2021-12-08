#
# source = inspect.getsource(train)
# parsed = ast.parse(source)

# for node in ast.walk(parsed):
#     if isinstance(node,ast.Call):
#         if isinstance(node.func, ast.Attribute):
#             if (node.func.value.id == 'layer'):
#                     if(node.func.attr == 'get_dataset'):
#                         print(ast.dump(node))
#                         print(node.args[0].value)


# ast.dump(parsed)
import pickle

import pandas as pd
import inspect
import os
import sys
import inspect
import ast
import cloudpickle

datasets = {}
models = {}

class Layer:
    entities = []
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

    def run(self, entities):
        self.entities = []
        for entity in entities:
            if not hasattr(entity, '_type'):
                raise Exception(f"Function {entity} is not decoratored!")
            elif entity._type == "dataset":
                self.entities.append(Dataset(entity))
            elif entity._type == "model":
                self.entities.append(Model(entity))

        print(f"--- Layer Infra: Running Project: {self.project_name} ---")

        self.setup()

        for entity in self.entities:
            entity.run()
        print(f"\n--- Layer Infra: Run Complete! ---")

    def get_dataset(self, name):
        for entity in self.entities:
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
            print(f'\nBuilding {Layer.entity_context}...')
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
            print(f'\nTraining {Layer.entity_context}...')
            res = func(*args, **kwargs)
            # TODO save returning entity to catalog
            return res

        wrapped._type = "model"
        wrapped._name = name
        source = inspect.getsource(func)

        return wrapped

    return inner
