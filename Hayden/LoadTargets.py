def LoadTargets(targetlist_fname):
    with open(targetlist_fname, 'r') as f:
        target_list = []
        for line in f:
            line = line.strip()  # Removes leading and trailing whitespace, including newlines
            if not line or line.startswith('#'):
                continue  
            target_list.append(line)
    return target_list