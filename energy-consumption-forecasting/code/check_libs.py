import sys

try:
    import statsmodels
    print("statsmodels is installed.")
except ImportError:
    print("statsmodels is NOT installed.")

try:
    import tensorflow as tf
    print("tensorflow is installed.")
except ImportError:
    try:
        import torch
        import torch.nn as nn
        print("pytorch is installed.")
    except ImportError:
        print("Neither tensorflow nor pytorch is installed.")
        
try:
    import sklearn
    print("sklearn is installed.")
except ImportError:
    print("sklearn is NOT installed.")
