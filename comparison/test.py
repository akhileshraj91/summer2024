import sys, pprint
import importlib.util

print('python executable:', sys.executable)
pprint.pprint(sys.path)
print('pandas spec:', importlib.util.find_spec('pandas'))