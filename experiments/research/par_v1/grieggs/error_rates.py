import logging

import editdistance


def cer(r, h, space_char=u' '):
    #Remove any double or trailing
    r = space_char.join(r.split())
    h = space_char.join(h.split())

    logging.debug('cer: r = `%s`; h = `%s`', r, h)

    return err(r, h)

def err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    return float(dis) / float(len(r))

def wer(r, h):
    r = r.split()
    h = h.split()

    return err(r, h)
