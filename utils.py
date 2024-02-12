# LOGGING FUNCTION WRITTEN BY M. PERRIN AND TAKEN FROM MCFOST UTILS.PY CODE! 
# https://github.com/mperrin/mcfost-python

import logging

def setup_logging(level='INFO',  filename=None, verbose=False):
    """ Simple wrapper function to set up convenient log defaults, for
    users not familiar with Python's logging system.

    """
    import logging
    _log = logging.getLogger('mcfost')

    lognames=['mcfost']

    if level.upper() =='NONE':
        # disable logging
        lev = logging.CRITICAL  # we don't generate any CRITICAL flagged log items, so
                                # setting the level to this is effectively the same as ignoring
                                # all log events. FIXME there's likely a cleaner way to do this.
        if verbose: print("No log messages will be shown.")
    else:
        lev = logging.__dict__[level.upper()] # obtain one of the DEBUG, INFO, WARN, or ERROR constants
        if verbose: print("Log messages of level {0} and above will be shown.".format(level))

    for name in lognames:
        logging.getLogger(name).setLevel(lev)
        _log.debug("Set log level of {name} to {lev}".format(name=name, lev=level.upper()))

    # set up screen logging
    logging.basicConfig(level=logging.INFO,format='%(name)-10s: %(levelname)-8s %(message)s')
    if verbose: print("Setup_logging is adjusting Python logging settings.")


    if str(filename).strip().lower() != 'none':
        hdlr = logging.FileHandler(filename)

        formatter = logging.Formatter('%(asctime)s %(name)-10s: %(levelname)-8s %(message)s')
        hdlr.setFormatter(formatter)

        for name in lognames:
            logging.getLogger(name).addHandler(hdlr)

        if verbose: print("Log outputs will also be saved to file "+filename)
        _log.debug("Log outputs will also be saved to file "+filename)