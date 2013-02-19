def read_token_info(filename):
    current_seq = []
    for line in (l.strip() for l in open(filename)):
        if not line:
            yield current_seq
            current_seq = []
        else:
            current_seq.append(line.split('\t'))
