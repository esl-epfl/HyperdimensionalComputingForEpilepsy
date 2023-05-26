'''
for loading all parameters
'''

import os, sys, getopt, inspect
from distutils.util import strtobool
import importlib


def strToType(target, arg):
    if type(target) == str:
        return arg
    elif type(target) == int:
        return int(arg)
    elif type(target) == float:
        return float(arg)
    elif type(target) == bool:
        return bool(strtobool(arg))
    elif type(target) == list:
        return [type(target[0])(i) for i in arg.replace('[', '').replace(']', '').split(',')]
    elif type(target) == enum:
        if strToType(target.val, arg) in target.allowed_vals:
            return strToType(target.val, arg)
        else:
            raise TypeError(f'Unrecognized parameter {arg}')
    else:
        raise TypeError('Unrecognized parameter type: %s' % type(target))


class enum:
    def __init__(self, val, vals):
        self.val = val
        self.allowed_vals = vals

    def __call__(self):
        return self.val


class Params:
    ignoreList = []

    def __init__(self, args):
        for arg in list(args.keys()):
            if arg in vars(self):
                vars(self)[arg] = strToType(vars(self)[arg], args.pop(arg))
        for key, val in vars(self).items():
            if type(val) == enum:
                vars(self)[key] = val.val

    def ignore(self, v):
        self.ignoreList.append(v)
        return v

    def __call__(self):
        return vars(self)

    def returnPath(self):
        paramStrings = ['{}:{}'.format(k, v) for k, v in vars(self).items() if v not in self.ignoreList]
        return os.path.join(*paramStrings) + '/' if paramStrings else '/'


paramModules = []


def getArgs(paramModuleList):
    for paramModule in paramModuleList:
        importlib.import_module(paramModule)
    a = [c[1]({})() for m in paramModules for c in inspect.getmembers(m, inspect.isclass) if c[1] != enum]
    defaultParams = {k: v for element in a for k, v in element.items()}

    # Extract all passed arguments
    optList, _ = getopt.getopt(sys.argv[1:], '', [s + '=' for s in defaultParams.keys()])
    optList = {k[2:]: v for (k, v) in optList}

    # Generate all parameter classes with passed arguments or default arguments
    paramsList = [param for paramList in [a.generateParams(optList) for a in paramModules] for param in paramList]
    a = dict()
    [a.update(b.__dict__) for b in paramsList]
    return a