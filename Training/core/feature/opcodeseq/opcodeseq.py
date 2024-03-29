from androguard.misc import AnalyzeAPK
from ...tools import utils

from ...config import logging

logger = logging.getLogger('feature.opcodeseq')


def get_opcode_sequences(apk_path, save_path):
    _1, _2, dx = AnalyzeAPK(apk_path)

    opcode_chunks = []
    for method in dx.get_methods():
        if method.is_external():
            continue
        mth_body = method.get_method()
        sequence = []
        for ins in mth_body.get_instructions():
            opcode = ins.get_op_value()
            if opcode < 0:
                opcode = 0
            elif opcode >= 256:
                opcode = 0
            else:
                opcode = opcode
            sequence.append(opcode)  # list of 'int'
        if len(sequence) > 0:
            opcode_chunks.append(sequence)
    dump_opcode(opcode_chunks, save_path)

    return save_path


def dump_opcode(opcode_chunks, save_path):
    utils.dump_json(opcode_chunks, save_path)
    return


def read_opcode(save_path):
    return utils.load_json(save_path)


def read_opcode_wrapper(save_path):
    try:
        return read_opcode(save_path)
    except Exception as e:
        return e


def feature_extr_wrapper(*args):
    """
    A helper function to catch the exception
    :param element: argurments for feature extraction
    :return: feature or Exception
    """
    try:
        return get_opcode_sequences(*args)
    except Exception as e:
        return e
