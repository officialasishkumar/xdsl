from __future__ import annotations

from collections.abc import Sequence
from enum import auto
from typing import ClassVar, Self

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    I64,
    AffineMapAttr,
    DenseArrayBase,
    FlatSymbolRefAttrConstr,
    FunctionType,
    IndexType,
    IntegerAttr,
    MemRefType,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.dialects.utils import EnumAttribute, parse_assignment, print_assignment
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Dialect,
    OperandArg,
    ParametrizedAttribute,
    Region,
    SpacedOpaqueSyntaxAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    NoTerminator,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.str_enum import StrEnum


@irdl_attr_definition
class AsyncTokenType(ParametrizedAttribute, TypeAttribute):
    name = "gpu.async.token"


class AllReduceOpEnum(StrEnum):
    Add = auto()
    And = auto()
    Max = auto()
    Min = auto()
    Mul = auto()
    Or = auto()
    Xor = auto()


class DimensionEnum(StrEnum):
    X = auto()
    Y = auto()
    Z = auto()


class ProcessorEnum(StrEnum):
    Sequential = auto()
    Block_X = auto()
    Block_Y = auto()
    Block_Z = auto()
    Thread_X = auto()
    Thread_Y = auto()
    Thread_Z = auto()


@irdl_attr_definition
class AllReduceOpAttr(EnumAttribute[AllReduceOpEnum], SpacedOpaqueSyntaxAttribute):
    name = "gpu.all_reduce_op"


@irdl_attr_definition
class DimensionAttr(EnumAttribute[DimensionEnum], SpacedOpaqueSyntaxAttribute):
    name = "gpu.dim"


@irdl_attr_definition
class ProcessorAttr(EnumAttribute[ProcessorEnum], SpacedOpaqueSyntaxAttribute):
    name = "gpu.processor"


@irdl_attr_definition
class LoopDimMapAttr(ParametrizedAttribute):
    name = "gpu.loop_dim_map"

    processor: ProcessorAttr
    map: AffineMapAttr
    bound: AffineMapAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("processor = ")
            printer.print_string(self.processor.data)
            printer.print_string(", map = ")
            printer.print_string(str(self.map.data))
            printer.print_string(", bound = ")
            printer.print_string(str(self.bound.data))

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            parser.parse_keyword("processor")
            parser.parse_punctuation("=")
            proc = ProcessorAttr.parse_parameter(parser)
            processor = ProcessorAttr(proc)
            parser.parse_punctuation(",")
            parser.parse_keyword("map")
            parser.parse_punctuation("=")
            map = AffineMapAttr(parser.parse_affine_map())
            parser.parse_punctuation(",")
            parser.parse_keyword("bound")
            parser.parse_punctuation("=")
            bound = AffineMapAttr(parser.parse_affine_map())
        return [processor, map, bound]


@irdl_op_definition
class AllocOp(IRDLOperation):
    name = "gpu.alloc"
    hostShared = opt_prop_def(UnitAttr)
    asyncDependencies = var_operand_def(AsyncTokenType)
    dynamicSizes = var_operand_def(IndexType)
    symbolOperands = var_operand_def(IndexType)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    result = result_def(memref.MemRefType)
    asyncToken = opt_result_def(AsyncTokenType)

    def verify_(self) -> None:
        ndyn = len(self.dynamicSizes)
        res_type = self.result.type
        assert isinstance(res_type, memref.MemRefType)
        ndyn_type = len([i for i in res_type.get_shape() if i == DYNAMIC_INDEX])
        if ndyn != ndyn_type:
            raise VerifyException(
                f"Expected {ndyn_type} dynamic sizes, got {ndyn}. All "
                "dynamic sizes need to be set in the alloc operation."
            )

    def __init__(
        self,
        return_type: memref.MemRefType,
        dynamic_sizes: Sequence[OperandArg] | None = None,
        host_shared: bool = False,
        async_dependencies: Sequence[OperandArg] | None = None,
        is_async: bool = False,
    ):
        token_return = [AsyncTokenType()] if is_async else []
        dynamic_sizes_vals: list[SSAValue] = (
            [SSAValue.get(e) for e in dynamic_sizes] if dynamic_sizes else []
        )
        async_dependencies_vals: list[SSAValue] = (
            [SSAValue.get(e) for e in async_dependencies] if async_dependencies else []
        )
        properties: dict[str, Attribute] = (
            {"hostShared": UnitAttr()} if host_shared else {}
        )
        super().__init__(
            operands=[async_dependencies_vals, dynamic_sizes_vals, []],
            result_types=[return_type, token_return],
            properties=properties,
        )


@irdl_op_definition
class AllReduceOp(IRDLOperation):
    name = "gpu.all_reduce"
    op = opt_prop_def(AllReduceOpAttr)
    uniform = opt_prop_def(UnitAttr)
    operand = operand_def(Attribute)
    result = result_def(Attribute)
    body = region_def()

    traits = traits_def(IsolatedFromAbove())

    @staticmethod
    def from_op(
        op: AllReduceOpAttr,
        operand: OperandArg,
        uniform: UnitAttr | None = None,
    ):
        return AllReduceOp.build(
            operands=[operand],
            result_types=[SSAValue.get(operand).type],
            properties={
                "op": op,
                "uniform": uniform,
            },
            regions=[Region()],
        )

    @staticmethod
    def from_body(body: Region, operand: OperandArg, uniform: UnitAttr | None = None):
        return AllReduceOp.build(
            operands=[operand],
            result_types=[SSAValue.get(operand).type],
            properties={"uniform": uniform} if uniform is not None else {},
            regions=[body],
        )

    def verify_(self) -> None:
        if self.result.type != self.operand.type:
            raise VerifyException(
                f"Type mismatch: result type is {self.result.type}, operand type is "
                f"{self.operand.type}. They must be the same type for gpu.all_reduce"
            )

        non_empty_body = bool(self.body.blocks)
        op_attr = self.op is not None
        if non_empty_body == op_attr:
            if op_attr:
                raise VerifyException(
                    "gpu.all_reduce can't have both a non-empty region and an op "
                    "attribute."
                )
            else:
                raise VerifyException(
                    "gpu.all_reduce need either a non empty body or an op attribute."
                )
        if non_empty_body:
            args_types = self.body.blocks[0].arg_types
            if args_types != (self.result.type, self.operand.type):
                raise VerifyException(
                    f"Expected {[str(t) for t in [self.result.type, self.operand.type]]}, "
                    f"got {[str(t) for t in args_types]}. A gpu.all_reduce's body must "
                    "have two arguments matching the result type."
                )


@irdl_op_definition
class BarrierOp(IRDLOperation):
    name = "gpu.barrier"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class BlockDimOp(IRDLOperation):
    name = "gpu.block_dim"
    dimension = prop_def(DimensionAttr)
    result = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class BlockIdOp(IRDLOperation):
    name = "gpu.block_id"
    dimension = prop_def(DimensionAttr)
    result = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class DeallocOp(IRDLOperation):
    name = "gpu.dealloc"

    asyncDependencies = var_operand_def(AsyncTokenType)
    buffer = operand_def(memref.MemRefType)

    irdl_options = (AttrSizedOperandSegments(),)

    asyncToken = opt_result_def(AsyncTokenType)

    def __init__(
        self,
        buffer: OperandArg,
        async_dependencies: Sequence[OperandArg] | None = None,
        is_async: bool = False,
    ):
        super().__init__(
            operands=[async_dependencies, buffer],
            result_types=[[AsyncTokenType()] if is_async else []],
        )


@irdl_op_definition
class MemcpyOp(IRDLOperation):
    name = "gpu.memcpy"

    asyncDependencies = var_operand_def(AsyncTokenType)
    dst = operand_def(memref.MemRefType)
    src = operand_def(memref.MemRefType)

    irdl_options = (AttrSizedOperandSegments(),)

    asyncToken = opt_result_def(AsyncTokenType)

    def __init__(
        self,
        source: OperandArg,
        destination: OperandArg,
        async_dependencies: Sequence[OperandArg] | None = None,
        is_async: bool = False,
    ):
        super().__init__(
            operands=[async_dependencies, destination, source],
            result_types=[[AsyncTokenType()] if is_async else []],
        )

    def verify_(self) -> None:
        if self.src.type != self.dst.type:
            raise VerifyException(
                f"Expected {self.dst.type}, got {self.src.type}. gpu.memcpy source and "
                "destination types must match."
            )


@irdl_op_definition
class ModuleOp(IRDLOperation):
    name = "gpu.module"

    body = region_def("single_block")
    sym_name = prop_def(SymbolNameConstraint())

    traits = traits_def(
        IsolatedFromAbove(),
        NoTerminator(),
        SymbolOpInterface(),
        SymbolTable(),
    )

    def __init__(self, name: SymbolRefAttr, body: Region):
        super().__init__(properties={"sym_name": name}, regions=[body])


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "gpu.func"

    body = region_def()
    sym_name = attr_def(SymbolNameConstraint())
    function_type = prop_def(FunctionType)
    kernel = opt_prop_def(UnitAttr)
    known_block_size = opt_attr_def(
        DenseArrayBase.constr(i32), attr_name="gpu.known_block_size"
    )
    known_grid_size = opt_attr_def(
        DenseArrayBase.constr(i32), attr_name="gpu.known_grid_size"
    )

    traits = traits_def(IsolatedFromAbove(), HasParent(ModuleOp), SymbolOpInterface())

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        kernel: bool | None = None,
        known_block_size: Sequence[int] | None = None,
        known_grid_size: Sequence[int] | None = None,
    ):
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        attributes: dict[str, Attribute | None] = {"sym_name": StringAttr(name)}
        properties: dict[str, Attribute | None] = {
            "function_type": function_type,
        }
        if known_block_size is not None:
            attributes["gpu.known_block_size"] = DenseArrayBase.from_list(
                i32, known_block_size
            )
        if known_grid_size is not None:
            attributes["gpu.known_grid_size"] = DenseArrayBase.from_list(
                i32, known_grid_size
            )
        if kernel:
            properties["kernel"] = UnitAttr()
        super().__init__(properties=properties, attributes=attributes, regions=[region])

    def verify_(self):
        entry_block: Block = self.body.blocks[0]
        function_inputs = self.function_type.inputs.data
        block_arg_types = entry_block.arg_types
        if function_inputs != block_arg_types:
            raise VerifyException(
                "Expected first entry block arguments to have the same types as the "
                "function input types"
            )
        if (self.kernel is not None) and (len(self.function_type.outputs) != 0):
            raise VerifyException("Expected void return type for kernel function")


@irdl_op_definition
class GlobalIdOp(IRDLOperation):
    name = "gpu.global_id"
    dimension = prop_def(DimensionAttr)
    result = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class GridDimOp(IRDLOperation):
    name = "gpu.grid_dim"
    dimension = prop_def(DimensionAttr)
    result = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class HostRegisterOp(IRDLOperation):
    """
    This op maps the provided host buffer into the device address space.

    This operation may not be supported in every environment, there is not yet a way to
    check at runtime whether this feature is supported.
    Writes from the host are guaranteed to be visible to device kernels that are launched
    afterwards. Writes from the device are guaranteed to be visible on the host after
    synchronizing with the device kernel completion.
    """

    name = "gpu.host_register"

    value = operand_def(memref.UnrankedMemRefType)

    def __init__(self, memref: OperandArg):
        super().__init__(operands=[SSAValue.get(memref)])


@irdl_op_definition
class HostUnregisterOp(IRDLOperation):
    """
    Unregisters a memref for access from device.
    """

    name = "gpu.host_unregister"

    value = operand_def(memref.UnrankedMemRefType)

    def __init__(self, memref: OperandArg):
        super().__init__(operands=[SSAValue.get(memref)])


@irdl_op_definition
class LaneIdOp(IRDLOperation):
    name = "gpu.lane_id"
    result = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class LaunchOp(IRDLOperation):
    """
    Launch a kernel on the specified grid of thread blocks. The body of the kernel is
    defined by the single region that this operation contains.

    The operation takes an optional list of async dependencies, the grid and block
    sizes along the x,y,z dimensions, optional cluster sizes along the x,y,z
    dimensions, and an optional amount of dynamic shared memory.

    The body region has at least twelve arguments, or eighteen if cluster sizes are
    present, grouped as follows:

    - three arguments that contain block identifiers along x,y,z dimensions;
    - three arguments that contain thread identifiers along x,y,z dimensions;
    - the grid and block sizes, as passed in the operands;
    - if cluster sizes are present, three arguments that contain cluster identifiers
      along x,y,z dimensions, followed by the cluster sizes;
    - a variadic number of workgroup memory attributions;
    - a variadic number of private memory attributions.

    The number of workgroup memory attributions is stored in the
    `workgroup_attributions` attribute. The optional `module` and `function`
    properties specify the module and the name of the kernel to outline the body to.

    See external [documentation](https://mlir.llvm.org/docs/Dialects/GPU/#gpulaunch-gpulaunchop).
    """

    name = "gpu.launch"
    asyncDependencies = var_operand_def(AsyncTokenType)
    gridSizeX = operand_def(IndexType)
    gridSizeY = operand_def(IndexType)
    gridSizeZ = operand_def(IndexType)
    blockSizeX = operand_def(IndexType)
    blockSizeY = operand_def(IndexType)
    blockSizeZ = operand_def(IndexType)
    clusterSizeX = opt_operand_def(IndexType)
    clusterSizeY = opt_operand_def(IndexType)
    clusterSizeZ = opt_operand_def(IndexType)
    dynamicSharedMemorySize = opt_operand_def(i32)
    module = opt_prop_def(FlatSymbolRefAttrConstr)
    function = opt_prop_def(FlatSymbolRefAttrConstr)
    num_workgroup_attributions = opt_attr_def(
        IntegerAttr[I64], attr_name="workgroup_attributions"
    )
    asyncToken = opt_result_def(AsyncTokenType)
    body = region_def()
    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    NUM_CONFIG_REGION_ARGS: ClassVar[int] = 12
    """
    The number of body arguments for the block and thread identifiers and the grid
    and block sizes, placed at the leading positions of the arguments.
    """

    NUM_CLUSTER_REGION_ARGS: ClassVar[int] = 6
    """
    The number of body arguments for the cluster identifiers and sizes, following
    the other configuration arguments when cluster sizes are present.
    """

    def __init__(
        self,
        body: Region,
        gridSize: Sequence[OperandArg],
        blockSize: Sequence[OperandArg],
        clusterSize: Sequence[OperandArg] = [],
        async_launch: bool = False,
        asyncDependencies: Sequence[OperandArg] | None = None,
        dynamicSharedMemorySize: OperandArg | None = None,
        *,
        num_workgroup_attributions: int = 0,
        module: str | SymbolRefAttr | None = None,
        function: str | SymbolRefAttr | None = None,
    ):
        if len(gridSize) != 3:
            raise ValueError(f"LaunchOp must have 3 gridSizes, got {len(gridSize)}")
        if len(blockSize) != 3:
            raise ValueError(f"LaunchOp must have 3 blockSizes, got {len(blockSize)}")
        if len(clusterSize) != 3 and len(clusterSize) != 0:
            raise ValueError(
                f"LaunchOp must have 0 or 3 clusterSizes, got {len(clusterSize)}"
            )
        operands = [
            (
                []
                if asyncDependencies is None
                else [SSAValue.get(a) for a in asyncDependencies]
            )
        ]

        operands += [SSAValue.get(gs) for gs in gridSize]
        operands += [SSAValue.get(bs) for bs in blockSize]
        if clusterSize:
            operands += [(SSAValue.get(cs),) for cs in clusterSize]
        else:
            operands += [(), (), ()]
        operands += [
            (
                []
                if dynamicSharedMemorySize is None
                else [SSAValue.get(dynamicSharedMemorySize)]
            )
        ]
        if isinstance(module, str):
            module = SymbolRefAttr(module)
        if isinstance(function, str):
            function = SymbolRefAttr(function)
        super().__init__(
            operands=operands,
            result_types=[[AsyncTokenType()] if async_launch else []],
            properties={"module": module, "function": function},
            attributes={
                "workgroup_attributions": IntegerAttr(num_workgroup_attributions, i64)
            },
            regions=[body],
        )

    @property
    def has_cluster_size(self) -> bool:
        """Whether the cluster sizes are specified."""
        return (
            self.clusterSizeX is not None
            and self.clusterSizeY is not None
            and self.clusterSizeZ is not None
        )

    @property
    def num_config_region_args(self) -> int:
        """
        The number of leading body arguments for the launch configuration, including
        the cluster identifiers and sizes if present.
        """
        if self.has_cluster_size:
            return self.NUM_CONFIG_REGION_ARGS + self.NUM_CLUSTER_REGION_ARGS
        return self.NUM_CONFIG_REGION_ARGS

    @property
    def workgroup_attributions(self) -> tuple[BlockArgument, ...]:
        """The body arguments for the buffers located in workgroup memory."""
        if self.num_workgroup_attributions is None:
            return ()
        start = self.num_config_region_args
        end = start + self.num_workgroup_attributions.value.data
        return self.body.blocks[0].args[start:end]

    @property
    def private_attributions(self) -> tuple[BlockArgument, ...]:
        """The body arguments for the buffers located in private memory."""
        start = self.num_config_region_args + len(self.workgroup_attributions)
        return self.body.blocks[0].args[start:]

    def verify_(self) -> None:
        cluster_sizes = (self.clusterSizeX, self.clusterSizeY, self.clusterSizeZ)
        if not self.has_cluster_size and any(s is not None for s in cluster_sizes):
            raise VerifyException("gpu.launch cluster sizes must all be present")
        if not any(b.ops for b in self.body.blocks):
            raise VerifyException("gpu.launch requires a non-empty body.")

        num_config_args = self.num_config_region_args
        num_workgroup_attributions = (
            0
            if self.num_workgroup_attributions is None
            else self.num_workgroup_attributions.value.data
        )
        if num_workgroup_attributions < 0:
            raise VerifyException(
                "Expected a non-negative number of workgroup attributions, got "
                f"{num_workgroup_attributions}"
            )
        args_type = self.body.blocks[0].arg_types
        config_args_type = args_type[:num_config_args]
        if (
            len(args_type) < num_config_args + num_workgroup_attributions
            or config_args_type != (IndexType(),) * num_config_args
        ):
            raise VerifyException(
                f"Expected {num_config_args} index arguments followed by "
                f"{num_workgroup_attributions} workgroup attributions, got "
                f"{[str(t) for t in args_type]}. gpu.launch's body arguments are 3 "
                "block identifiers, 3 thread identifiers, 3 grid sizes, 3 block sizes, "
                "then 3 cluster identifiers and 3 cluster sizes if cluster sizes are "
                "present, followed by the workgroup and private memory attributions"
            )

        for attribution in self.workgroup_attributions + self.private_attributions:
            if not isinstance(attribution.type, MemRefType):
                raise VerifyException(
                    f"Expected memref type in attribution, got {attribution.type}"
                )

    def print(self, printer: Printer) -> None:
        if self.asyncToken is not None:
            printer.print_string(" async")
        if self.asyncDependencies:
            printer.print_string(" [")
            printer.print_list(self.asyncDependencies, printer.print_operand)
            printer.print_string("]")

        args = self.body.blocks[0].args
        if self.has_cluster_size:
            cluster_sizes = tuple(
                size
                for size in (self.clusterSizeX, self.clusterSizeY, self.clusterSizeZ)
                if size is not None
            )
            printer.print_string(" clusters")
            _print_size_assignment(printer, args[12:15], args[15:18], cluster_sizes)
        printer.print_string(" blocks")
        _print_size_assignment(
            printer,
            args[0:3],
            args[6:9],
            (self.gridSizeX, self.gridSizeY, self.gridSizeZ),
        )
        printer.print_string(" threads")
        _print_size_assignment(
            printer,
            args[3:6],
            args[9:12],
            (self.blockSizeX, self.blockSizeY, self.blockSizeZ),
        )
        if self.dynamicSharedMemorySize is not None:
            printer.print_string(" dynamic_shared_memory_size ")
            printer.print_operand(self.dynamicSharedMemorySize)
        if self.module is not None:
            printer.print_string(" module(")
            printer.print_symbol_name(self.module.root_reference)
            printer.print_string(")")
        if self.function is not None:
            printer.print_string(" function(")
            printer.print_symbol_name(self.function.root_reference)
            printer.print_string(")")
        _print_attributions(printer, "workgroup", self.workgroup_attributions)
        _print_attributions(printer, "private", self.private_attributions)
        printer.print_string(" ")
        printer.print_region(
            self.body, print_entry_block_args=False, print_empty_block=False
        )
        printer.print_op_attributes(
            self.attributes, reserved_attr_names=("workgroup_attributions",)
        )

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        async_launch = parser.parse_optional_keyword("async") is not None
        async_dependencies = parser.parse_optional_comma_separated_list(
            Parser.Delimiter.SQUARE, lambda: parser.parse_operand(AsyncTokenType())
        )

        cluster_ids: Sequence[Parser.UnresolvedArgument] = ()
        cluster_size_args: Sequence[Parser.UnresolvedArgument] = ()
        cluster_sizes: Sequence[SSAValue] = ()
        if parser.parse_optional_keyword("clusters") is not None:
            cluster_ids, cluster_size_args, cluster_sizes = _parse_size_assignment(
                parser
            )
        parser.parse_keyword("blocks")
        block_ids, grid_size_args, grid_sizes = _parse_size_assignment(parser)
        parser.parse_keyword("threads")
        thread_ids, block_size_args, block_sizes = _parse_size_assignment(parser)

        dynamic_shared_memory_size = None
        if parser.parse_optional_keyword("dynamic_shared_memory_size") is not None:
            dynamic_shared_memory_size = parser.parse_operand(i32)

        module = None
        if parser.parse_optional_keyword("module") is not None:
            with parser.in_parens():
                module = SymbolRefAttr(parser.parse_symbol_name())
        function = None
        if parser.parse_optional_keyword("function") is not None:
            with parser.in_parens():
                function = SymbolRefAttr(parser.parse_symbol_name())

        workgroup_attributions = _parse_attributions(parser, "workgroup")
        private_attributions = _parse_attributions(parser, "private")

        config_args = (
            *block_ids,
            *thread_ids,
            *grid_size_args,
            *block_size_args,
            *cluster_ids,
            *cluster_size_args,
        )
        body = parser.parse_region(
            (
                *(arg.resolve(IndexType()) for arg in config_args),
                *workgroup_attributions,
                *private_attributions,
            )
        )
        attrs = parser.parse_optional_attr_dict_with_reserved_attr_names(
            ("workgroup_attributions",)
        )

        op = cls(
            body,
            grid_sizes,
            block_sizes,
            cluster_sizes,
            async_launch,
            async_dependencies,
            dynamic_shared_memory_size,
            num_workgroup_attributions=len(workgroup_attributions),
            module=module,
            function=function,
        )
        if attrs is not None:
            op.attributes |= attrs.data
        return op


def _print_size_assignment(
    printer: Printer,
    ids: Sequence[BlockArgument],
    size_args: Sequence[BlockArgument],
    sizes: Sequence[SSAValue],
) -> None:
    """
    Print a launch size assignment, with format:
      `(` ids `)` `in` `(` size-arg `=` size (`,` size-arg `=` size)* `)`
    """
    with printer.in_parens():
        printer.print_list(ids, printer.print_ssa_value)
    printer.print_string(" in ")
    with printer.in_parens():
        printer.print_list(
            zip(size_args, sizes), lambda pair: print_assignment(printer, *pair)
        )


def _parse_size_assignment(
    parser: Parser,
) -> tuple[
    list[Parser.UnresolvedArgument],
    list[Parser.UnresolvedArgument],
    list[SSAValue[IndexType]],
]:
    """
    Parse a launch size assignment, with format:
      `(` ids `)` `in` `(` size-arg `=` size (`,` size-arg `=` size)* `)`
    where there are exactly three identifiers and sizes, for the x, y and z
    dimensions.
    """
    pos = parser.pos
    ids = parser.parse_comma_separated_list(
        Parser.Delimiter.PAREN, lambda: parser.parse_argument(expect_type=False)
    )
    if len(ids) != 3:
        parser.raise_error(f"Expected 3 identifiers, got {len(ids)}", pos, parser.pos)
    parser.parse_keyword("in")
    pos = parser.pos
    assignments = parser.parse_comma_separated_list(
        Parser.Delimiter.PAREN, lambda: parse_assignment(parser)
    )
    if len(assignments) != 3:
        parser.raise_error(
            f"Expected 3 size assignments, got {len(assignments)}", pos, parser.pos
        )
    size_args = [arg for arg, _ in assignments]
    sizes = [parser.resolve_operand(size, IndexType()) for _, size in assignments]
    return ids, size_args, sizes


def _print_attributions(
    printer: Printer, keyword: str, attributions: Sequence[BlockArgument]
) -> None:
    """
    Print memory attributions, if any, with format:
      keyword `(` ssa-id `:` type (`,` ssa-id `:` type)* `)`
    """
    if not attributions:
        return
    printer.print_string(f" {keyword}")
    with printer.in_parens():
        printer.print_list(attributions, lambda arg: _print_attribution(printer, arg))


def _print_attribution(printer: Printer, attribution: BlockArgument) -> None:
    printer.print_ssa_value(attribution)
    printer.print_string(" : ")
    printer.print_attribute(attribution.type)


def _parse_attributions(parser: Parser, keyword: str) -> list[Parser.Argument]:
    """
    Parse memory attributions, if present, with format:
      keyword `(` ssa-id `:` type (`,` ssa-id `:` type)* `)`
    """
    if parser.parse_optional_keyword(keyword) is None:
        return []
    return parser.parse_comma_separated_list(
        Parser.Delimiter.PAREN, parser.parse_argument
    )


@irdl_op_definition
class LaunchFuncOp(IRDLOperation):
    """
    Launch a kernel function on the specified grid of thread blocks. gpu.launch
    operations are lowered to gpu.launch_func operations by outlining the kernel body
    into a function in a dedicated module, which reflects the separate compilation
    process. The kernel function is required to have the gpu.kernel attribute. The
    module containing the kernel function is required to be a gpu.module. And finally,
    the module containing the kernel module (which thus cannot be the top-level module)
    is required to have the gpu.container_module attribute. The gpu.launch_func operation
    has a symbol attribute named kernel to identify the fully specified kernel function
    to launch (both the gpu.module and func).

    The gpu.launch_func supports async dependencies: the kernel does not start executing
    until the ops producing those async dependencies have completed.

    By default, the host implicitly blocks until kernel execution has completed. If
    the async keyword is present, the host does not block but instead a !gpu.async.token
    is returned. Other async GPU ops can take this token as dependency.

    The operation requires at least the grid and block sizes along the x,y,z dimensions
    as arguments. When a lower-dimensional kernel is required, unused sizes must be
    explicitly set to 1.

    The remaining operands are optional. The first optional operand corresponds to the
    amount of dynamic shared memory a kernel's workgroup should be allocated; when this
    operand is not present, a zero size is assumed.

    The remaining operands if present are passed as arguments to the kernel function.
    """

    name = "gpu.launch_func"
    asyncDependencies = var_operand_def(AsyncTokenType)
    gridSizeX = operand_def(AnyOf.get(IndexType, i32, i64))
    gridSizeY = operand_def(AnyOf.get(IndexType, i32, i64))
    gridSizeZ = operand_def(AnyOf.get(IndexType, i32, i64))
    blockSizeX = operand_def(AnyOf.get(IndexType, i32, i64))
    blockSizeY = operand_def(AnyOf.get(IndexType, i32, i64))
    blockSizeZ = operand_def(AnyOf.get(IndexType, i32, i64))
    clusterSizeX = opt_operand_def(AnyOf.get(IndexType, i32, i64))
    clusterSizeY = opt_operand_def(AnyOf.get(IndexType, i32, i64))
    clusterSizeZ = opt_operand_def(AnyOf.get(IndexType, i32, i64))
    dynamicSharedMemorySize = opt_operand_def(i32)
    kernelOperands = var_operand_def()
    asyncObject = opt_operand_def()

    asyncToken = opt_result_def(AsyncTokenType)

    kernel = prop_def(SymbolRefAttr)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def __init__(
        self,
        func: SymbolRefAttr,
        gridSize: Sequence[OperandArg],
        blockSize: Sequence[OperandArg],
        clusterSize: Sequence[OperandArg] | None = None,
        kernelOperands: Sequence[OperandArg] | None = None,
        async_launch: bool = False,
        asyncDependencies: Sequence[OperandArg] | None = None,
        dynamicSharedMemorySize: OperandArg | None = None,
    ):
        if len(gridSize) != 3:
            raise ValueError(f"LaunchOp must have 3 gridSizes, got {len(gridSize)}")
        if len(blockSize) != 3:
            raise ValueError(f"LaunchOp must have 3 blockSizes, got {len(blockSize)}")
        clusterSizeOperands: Sequence[OperandArg | Sequence[OperandArg]]
        if clusterSize is None:
            clusterSizeOperands = [[], [], []]
        else:
            clusterSizeOperands = clusterSize
        if len(clusterSizeOperands) != 3:
            raise ValueError(
                f"LaunchFuncOp must have 3 cluterSizes if any, got {len(clusterSizeOperands)}"
            )

        super().__init__(
            operands=[
                asyncDependencies,
                *gridSize,
                *blockSize,
                *clusterSizeOperands,
                dynamicSharedMemorySize,
                kernelOperands,
                [],
            ],
            result_types=[[AsyncTokenType()] if async_launch else []],
            properties={"kernel": func},
        )


@irdl_op_definition
class NumSubgroupsOp(IRDLOperation):
    name = "gpu.num_subgroups"
    result = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "gpu.return"

    args = var_operand_def()

    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(self, operands: Sequence[OperandArg]):
        super().__init__(operands=[operands])


@irdl_op_definition
class SetDefaultDeviceOp(IRDLOperation):
    name = "gpu.set_default_device"
    devIndex = operand_def(i32)

    def __init__(self, devIndex: OperandArg):
        super().__init__(operands=[SSAValue.get(devIndex)])


@irdl_op_definition
class SubgroupIdOp(IRDLOperation):
    name = "gpu.subgroup_id"
    result = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class SubgroupSizeOp(IRDLOperation):
    name = "gpu.subgroup_size"
    result = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


@irdl_op_definition
class TerminatorOp(IRDLOperation):
    name = "gpu.terminator"

    traits = traits_def(HasParent(LaunchOp), IsTerminator())

    assembly_format = "attr-dict"

    def __init__(self):
        super().__init__()


@irdl_op_definition
class ThreadIdOp(IRDLOperation):
    name = "gpu.thread_id"
    dimension = prop_def(DimensionAttr)
    result = result_def(IndexType)

    def __init__(self, dim: DimensionAttr):
        super().__init__(result_types=[IndexType()], properties={"dimension": dim})


@irdl_op_definition
class WaitOp(IRDLOperation):
    name = "gpu.wait"
    asyncDependencies = var_operand_def(AsyncTokenType)
    asyncToken = opt_result_def(AsyncTokenType)

    def __init__(
        self,
        async_dependencies: Sequence[OperandArg] | None = None,
    ):
        super().__init__(
            operands=[async_dependencies],
            result_types=[[AsyncTokenType()]],
        )


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "gpu.yield"
    values = var_operand_def(Attribute)

    def __init__(self, operands: Sequence[OperandArg]):
        super().__init__(operands=[operands])

    traits = traits_def(IsTerminator())

    def verify_(self) -> None:
        op = self.parent_op()
        if op is not None:
            yield_type = self.values.types
            result_type = op.result_types
            if yield_type != result_type:
                raise VerifyException(
                    f"Expected {[str(t) for t in result_type]}, got {[str(t) for t in yield_type]}. The gpu.yield values "
                    "types must match its enclosing operation result types."
                )


GPU = Dialect(
    "gpu",
    [
        AllocOp,
        AllReduceOp,
        BarrierOp,
        BlockDimOp,
        BlockIdOp,
        DeallocOp,
        FuncOp,
        GlobalIdOp,
        GridDimOp,
        HostRegisterOp,
        HostUnregisterOp,
        LaneIdOp,
        LaunchOp,
        LaunchFuncOp,
        MemcpyOp,
        ModuleOp,
        NumSubgroupsOp,
        ReturnOp,
        SetDefaultDeviceOp,
        SubgroupIdOp,
        SubgroupSizeOp,
        TerminatorOp,
        ThreadIdOp,
        WaitOp,
        YieldOp,
    ],
    [
        AsyncTokenType,
        AllReduceOpAttr,
        DimensionAttr,
        ProcessorAttr,
        LoopDimMapAttr,
    ],
)
