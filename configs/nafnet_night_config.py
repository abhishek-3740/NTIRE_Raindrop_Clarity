import os
import sys
import yaml

# 1. Locate the NIGHT YAML file
yaml_file = os.path.join(os.path.dirname(__file__), "nafnet_night_config.yaml")

if not os.path.exists(yaml_file):
    raise FileNotFoundError(f"Config file not found at: {yaml_file}")

# 2. Load the YAML data
with open(yaml_file, "r") as f:
    config_dict = yaml.safe_load(f)

# 3. MAGIC STEP: Inject settings into this module
this_module = sys.modules[__name__]

for key, value in config_dict.items():
    setattr(this_module, key, value)