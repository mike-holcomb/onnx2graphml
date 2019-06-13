from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import os

import networkx as nx
from typing import Text, Sequence, Any, Optional, Dict, Union, TypeVar, Callable, Tuple, List, cast


def _sanitize_str(s):  # type: (Union[Text, bytes]) -> Text
    """
    Directly lifted from here:
    https://github.com/onnx/onnx/blob/dd599b05f424eb161a31f3e059566a33310dbe5e/onnx/helper.py#L343
    """

    if isinstance(s, text_type):
        sanitized = s
    elif isinstance(s, binary_type):
        sanitized = s.decode('utf-8', errors='ignore')
    else:
        sanitized = str(s)
    if len(sanitized) < 64:
        return sanitized
    else:
        return sanitized[:64] + '...<+len=%d>' % (len(sanitized) - 64)


def networkxable_attribute(attr, subgraphs=False):  # type: (AttributeProto, bool) -> Union[Text, Tuple[Text, List[GraphProto]]]
    """
    Adapted from onnx.helper.printable_attribute()
    https://github.com/onnx/onnx/blob/dd599b05f424eb161a31f3e059566a33310dbe5e/onnx/helper.py#L356
    """
    key = attr.name
    value = []

    def str_float(f):  # type: (float) -> Text
        # NB: Different Python versions print different numbers of trailing
        # decimals, specifying this explicitly keeps it consistent for all
        # versions
        return '{:.15g}'.format(f)

    def str_int(i):  # type: (int) -> Text
        # NB: In Python 2, longs will repr() as '2L', which is ugly and
        # unnecessary.  Explicitly format it to keep it consistent.
        return '{:d}'.format(i)

    def str_str(s):  # type: (Text) -> Text
        return repr(s)

    _T = TypeVar('_T')  # noqa

    def str_list(str_elem, xs):  # type: (Callable[[_T], Text], Sequence[_T]) -> Tuple[Text, Text]
        return '[' + ', '.join(map(str_elem, xs)) + ']'

    # for now, this logic should continue to work as long as we are running on a proto3
    # implementation. If/when we switch to proto3, we will need to use attr.type

    # To support printing subgraphs, if we find a graph attribute, print out
    # its name here and pass the graph itself up to the caller for later
    # printing.

    if attr.HasField("f"):
        value.append(str_float(attr.f))
    elif attr.HasField("i"):
        value.append(str_int(attr.i))
    elif attr.HasField("s"):
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        value.append(repr(_sanitize_str(attr.s)))
    elif attr.HasField("t"):
        if len(attr.t.dims) > 0:
            value.append("<Tensor>")
        else:
            # special case to print scalars
            field = STORAGE_TENSOR_TYPE_TO_FIELD[attr.t.data_type]
            value.append('<Scalar Tensor {}>'.format(str(getattr(attr.t, field))))
    #     elif attr.HasField("g"):
    #         value.append("<graph {}>".format(attr.g.name))
    #         graphs.append(attr.g)
    elif attr.floats:
        value.append(str_list(str_float, attr.floats))
    elif attr.ints:
        value.append(str_list(str_int, attr.ints))
    elif attr.strings:
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        value.append(str(list(map(_sanitize_str, attr.strings))))
    elif attr.tensors:
        value.append("[<Tensor>, ...]")

    #     elif attr.graphs:
    #         content.append('[')
    #         for i, g in enumerate(attr.graphs):
    #             comma = ',' if i != len(attr.graphs) - 1 else ''
    #             content.append('<graph {}>{}'.format(g.name, comma))
    #         content.append(']')
    #         graphs.extend(attr.graphs)
    else:
        value.append("<Unknown>")
    #     if subgraphs:
    #         return ' '.join(value), graphs
    #     else:
    return key, ' '.join(value)


def _get_out_name(node):
    if len(node.output) > 0:
        outputs = [str(out_name) for out_name in node.output]
    else:
        outputs = [node.output]

    return outputs[0]


def _get_node_name(node):
    return node.name if len(node.name) > 0 else _get_out_name(node)


def networkxable_node(node, prefix='',
                      subgraphs=False):  # type: (NodeProto, Text, bool) -> Union[Text, Tuple[Text, List[GraphProto]]]
    '''
    Adapted from onnx.helper.printable_graph()
    https://github.com/onnx/onnx/blob/dd599b05f424eb161a31f3e059566a33310dbe5e/onnx/helper.py#L455
    '''
    #     content = []
    #     if len(node.output):
    #         content.append(
    #             ', '.join(['%{}'.format(name) for name in node.output]))
    #         content.append('=')
    #     # To deal with nested graphs
    #     graphs = []  # type: List[GraphProto]

    attrs = {}
    for attr in node.attribute:
        #         if subgraphs:
        #             printed_attr, gs = printable_attribute(attr, subgraphs)
        #             assert isinstance(gs, list)
        #             graphs.extend(gs)
        #             printed_attrs.append(printed_attr)
        #         else:
        k, v = networkxable_attribute(attr)
        assert isinstance(v, Text)
        attrs[k] = v

    attrs['op_type'] = node.op_type

    node_name = _get_node_name(node)

    #     printed_attributes = ', '.join(sorted(printed_attrs))
    #     printed_inputs = ', '.join(['%{}'.format(name) for name in node.input])

    #     if node.attribute:
    #         content.append("{}[{}]({})".format(node.op_type, printed_attributes, printed_inputs))
    #     else:
    #         content.append("{}({})".format(node.op_type, printed_inputs))

    #     if subgraphs:
    #         return prefix + ' '.join(content), graphs
    #     else:
    return node_name, attrs


def networkxable_graph(graph, prefix=''):  # type: (GraphProto, Text) -> Text
    '''
    Adapted from onnx.helper.printable_graph()
    https://github.com/onnx/onnx/blob/dd599b05f424eb161a31f3e059566a33310dbe5e/onnx/helper.py#L486
    '''
    DG = nx.DiGraph()
    initialized = {t.name for t in graph.initializer}

    if len(graph.input):
        in_strs = []
        init_strs = []
        for inp in graph.input:
            if inp.name not in initialized:
                DG.add_node(inp.name)
    #             else:
    #                 DG.add_node(inp.name)
    #         if in_strs:
    #             content.append(prefix + ' '.join(header))
    #             header = []
    #             for line in in_strs:
    #                 content.append(prefix + '  ' + line)
    #         header.append(")")

    #         if init_strs:
    #             header.append("initializers (")
    #             content.append(prefix + ' '.join(header))
    #             header = []
    #             for line in init_strs:
    #                 content.append(prefix + '  ' + line)
    #             header.append(")")

    #     header.append('{')
    #     content.append(prefix + ' '.join(header))
    #     graphs = []  # type: List[GraphProto]
    #     # body

    # ADD NODES
    for node in graph.node:
        nname, nattrs = networkxable_node(node)
        DG.add_node(nname, **nattrs)

    # ADD EDGES
    for node in graph.node:
        for innode in node.input:
            if innode not in initialized:
                DG.add_edge(innode, _get_node_name(node))

    #         content.append(pn)
    #         graphs.extend(gs)
    #     # tail
    #     tail = ['return']
    #     if len(graph.output):
    #         tail.append(
    #             ', '.join(['%{}'.format(out.name) for out in graph.output]))
    #     content.append(indent + ' '.join(tail))
    #     # closing bracket
    #     content.append(prefix + '}')
    #     for g in graphs:
    #         DG.append('\n' + printable_graph(g))
    return DG


def convert(onnx_file, output_file='onnx.graphml'):
    onnx_model = onnx.load(onnx_file)
    g = networkxable_graph(onnx_model.graph)
    nx.write_graphml(g, output_file)
