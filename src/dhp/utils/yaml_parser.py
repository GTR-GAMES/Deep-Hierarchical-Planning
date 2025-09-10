import yaml

def convert_types(value):
    if isinstance(value, str):
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Convert boolean strings
        if value.lower() in ('true', 'yes', 'on'):
            return True
        elif value.lower() in ('false', 'no', 'off'):
            return False
        elif value.lower() in ('null', 'none'):
            return None
    
    return value

def parse_yaml_with_types(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Recursively convert all string values
    if isinstance(data, dict):
        return {k: convert_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_types(item) for item in data]
    else:
        return convert_types(data)