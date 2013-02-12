from utils import read_token_info

def read_preterminal_data(filename):
    def transform_ident(t, p, id_type, is_dec):
        is_dec = True if is_dec == 'true' else False
        token = id_type + '_IDENTIFIER'
        if is_dec:
            token += '_DEC'
        return token
    
    for seq in read_token_info(filename):
        cur_seq = []
        for t, p, id_type, is_dec in seq:
            if p == 'Identifier':
                p = transform_ident(t, p, id_type, is_dec)
            cur_seq.append(p)
        yield ' '.join(cur_seq)

def read_baseline_preterminal_data(filename):
    def transform_ident(t, p, id_type, is_dec):
        return id_type + '_IDENTIFIER'
    
    for seq in read_token_info(filename):
        cur_seq = []
        for t, p, id_type, is_dec in seq:
            if p == 'Identifier':
                p = transform_ident(t, p, id_type, is_dec)
            cur_seq.append(p)
        yield ' '.join(cur_seq)
