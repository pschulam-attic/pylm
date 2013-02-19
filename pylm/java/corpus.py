def read(stream):
    preterminal_seqs = []
    identifiers = []

    cur_seq = []
    for line in (l.strip() for l in stream):
        if not line:
            preterminal_seqs.append(cur_seq)
            cur_seq = []
        else:
            t, p, id_type, is_dec = line.split('\t')
            cur_seq.append(p)
            if p == 'Identifier':
                identifiers.append(t)

    return preterminal_seqs, identifiers
