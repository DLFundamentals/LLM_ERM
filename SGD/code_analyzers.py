import ast

def count_function_characters(file_path, func_name):
    # Read the script from the file
    with open(file_path, 'r') as file:
        script = file.read()
    
    # Parse the script into an AST
    tree = ast.parse(script)
    
    # Find the function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Extract the function source code from the script
            function_code = ast.get_source_segment(script, node)
            # Return the length of the function code in characters
            return len(function_code)
    
    # Return None if the function is not found
    return None