// RUN: xdsl-opt %s --verify-diagnostics --parsing-diagnostics --split-input-file | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        ^bb(%lhs: index, %rhs: index):
            %sum = "arith.addi"(%lhs, %rhs) : (index, index) -> index
            "gpu.yield"(%sum) : (index) -> ()
        }) {"op" = #gpu<all_reduce_op add>} : (index) -> index
    }) {"sym_name" = "all_reduce_both"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.all_reduce can't have both a non-empty region and an op attribute.

// -----

"builtin.module"() ({
}) {"wrong_all_reduce_operation" = #gpu<all_reduce_op magic>}: () -> ()

// CHECK: Expected `add`, `and`, `max`, `min`, `mul`, `or`, or `xor`.

// -----

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        ^bb(%lhs: index):
            %c = "arith.constant"() {"value" = 42 : index} : () -> index
            %sum = "arith.addi"(%lhs, %c) : (index, index) -> index
            "gpu.yield"(%sum) : (index) -> ()
        }) : (index) -> index
    }) {"sym_name" = "all_reduce_body_types"} : () -> ()
}) {} : () -> ()

// CHECK: Expected ['index', 'index'], got ['index']. A gpu.all_reduce's body must have two arguments matching the result type.

// -----

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({}) : (index) -> index
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.all_reduce need either a non empty body or an op attribute.

// -----

"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        }) {"op" = #gpu<all_reduce_op add>} : (index) -> f32
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: Type mismatch: result type is f32, operand type is index. They must be the same type for gpu.all_reduce

// -----

"builtin.module"() ({
    %0 = "arith.constant"() {"value" = 10 : index} : () -> index
    %gdmemref = "gpu.alloc"(%0, %0, %0) {operandSegmentSizes = array<i32: 0, 3, 0>} : (index, index, index) -> memref<10x10x10xf64>
}) : () -> ()

// CHECK: Expected 0 dynamic sizes, got 3. All dynamic sizes need to be set in the alloc operation.

// -----

"builtin.module"() ({
}) {"wrong_dim" = #gpu<dim w>}: () -> ()

// CHECK: Expected `x`, `y`, or `z`.

// -----

"builtin.module"()({
    "gpu.module"()({
        %n = "arith.constant"() {"value" = 13 : index} : () -> index
        %one = "arith.constant"() {"value" = 1 : index} : () -> index

        "gpu.launch"(%one, %one, %n, %one, %one, %one) ({
        ^bb0(%bx : index, %by : index, %bz : index,
            %tx : index, %ty: index, %tz: index):
            %sum = "gpu.all_reduce"(%tx) ({
            }) {"op" = #gpu<all_reduce_op add>} : (index) -> index
            %final = "arith.muli"(%sum, %one) : (index, index) -> index
            "gpu.terminator"() : () -> ()
        }) {operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>} : (index, index, index, index, index, index) -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: Expected 12 index arguments followed by 0 workgroup attributions, got ['index', 'index', 'index', 'index', 'index', 'index']. gpu.launch's body arguments are 3 block identifiers, 3 thread identifiers, 3 grid sizes, 3 block sizes, then 3 cluster identifiers and 3 cluster sizes if cluster sizes are present, followed by the workgroup and private memory attributions

// -----

"builtin.module"()({
    "gpu.module"()({
        %n = "arith.constant"() {"value" = 13 : index} : () -> index
        %one = "arith.constant"() {"value" = 1 : index} : () -> index

        "gpu.launch"(%one, %one, %n, %one, %one, %one) ({})
        {operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>} : (index, index, index, index, index, index) -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.launch requires a non-empty body.

// -----

%one = "test.op"() : () -> index
"gpu.launch"(%one, %one, %one, %one, %one, %one, %one) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0>}> ({
^bb0(%bx : index, %by : index, %bz : index, %tx : index, %ty : index, %tz : index,
     %gx : index, %gy : index, %gz : index, %sx : index, %sy : index, %sz : index):
  gpu.terminator
}) : (index, index, index, index, index, index, index) -> ()

// CHECK: gpu.launch cluster sizes must all be present

// -----

%one = "test.op"() : () -> index
gpu.launch clusters(%cx, %cy, %cz) in (%scx = %one, %scy = %one, %scz = %one)
    blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
    threads(%tx, %ty, %tz) in (%sx = %one, %sy = %one, %sz = %one)
    workgroup(%workgroup : i32) {
  gpu.terminator
}

// CHECK: Expected memref type in attribution, got i32

// -----

%one = "test.op"() : () -> index
"gpu.launch"(%one, %one, %one, %one, %one, %one) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
^bb0(%bx : index, %by : index, %bz : index, %tx : index, %ty : index, %tz : index,
     %gx : index, %gy : index, %gz : index, %sx : index, %sy : index, %sz : index):
  gpu.terminator
}) {workgroup_attributions = 1 : i64} : (index, index, index, index, index, index) -> ()

// CHECK: Expected 12 index arguments followed by 1 workgroup attributions, got

// -----

%one = "test.op"() : () -> index
"gpu.launch"(%one, %one, %one, %one, %one, %one) <{operandSegmentSizes = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>}> ({
^bb0(%bx : index, %by : index, %bz : index, %tx : index, %ty : index, %tz : index,
     %gx : index, %gy : index, %gz : index, %sx : index, %sy : index, %sz : index):
  gpu.terminator
}) {workgroup_attributions = -1 : i64} : (index, index, index, index, index, index) -> ()

// CHECK: Expected a non-negative number of workgroup attributions, got -1

// -----

%one = "test.op"() : () -> index
gpu.launch blocks(%bx, %by) in (%gx = %one, %gy = %one, %gz = %one)
    threads(%tx, %ty, %tz) in (%sx = %one, %sy = %one, %sz = %one) {
  gpu.terminator
}

// CHECK: Expected 3 identifiers, got 2

// -----

%one = "test.op"() : () -> index
gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
    threads(%tx, %ty, %tz) in (%sx = %one, %sy = %one) {
  gpu.terminator
}

// CHECK: Expected 3 size assignments, got 2

// -----

%one = "test.op"() : () -> index
gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
    threads(%tx, %ty, %tz) in (%sx = %one, %sy = %one, %sz = %one) {
  gpu.terminator
} {workgroup_attributions = 0 : i64}

// CHECK: Attribute dictionary entry 'workgroup_attributions' is already passed through the operation custom assembly format

// -----

"builtin.module"() ({
    %memref = "memref.alloc"() {operandSegmentSizes = array<i32: 0, 0>} : () -> memref<10x10xi32>
    %ten = "arith.constant"() {"value" = 10 : index} : () -> index
    %gmemref = "gpu.alloc"(%ten, %ten) {operandSegmentSizes = array<i32: 0, 2, 0>} : (index, index) -> memref<?x?xi32>

    "gpu.memcpy"(%memref, %gmemref) {operandSegmentSizes = array<i32: 0, 1, 1>} : (memref<10x10xi32>, memref<?x?xi32>) -> ()

}) : () -> ()

// CHECK: Expected memref<10x10xi32>, got memref<?x?xi32>. gpu.memcpy source and destination types must match.

// -----

"builtin.module"()({
    "gpu.module"()({
^bb0:
    }) {} : () -> ()
}) {} : () -> ()

// CHECK: property 'sym_name' expected in operation 'gpu.module'

// -----


"builtin.module"()({
    "gpu.module"()({
        %init = "arith.constant"() {"value" = 42 : index} : () -> index
        %sum = "gpu.all_reduce"(%init) ({
        ^bb(%lhs: index, %rhs: index):
            %sum = "arith.addi"(%lhs, %rhs) : (index, index) -> index
            %float = "arith.constant"() {"value" = 42.0 : f32} : () -> f32
            "gpu.yield"(%float) : (f32) -> ()
        }) {} : (index) -> index
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: Expected ['index'], got ['f32']. The gpu.yield values types must match its enclosing operation result types.

// -----

"builtin.module"()({
    "gpu.module"()({
        "gpu.func"() ({
        ^bb0(%arg0: index):
            "gpu.return"() : () -> ()
        }) {"sym_name" = "foo", "kernel", "function_type" = () -> ()} : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) : () -> ()

// CHECK: Expected first entry block arguments to have the same types as the function input types

// -----

"builtin.module"()({
    "gpu.module"()({
        "gpu.func"() ({
        ^bb0(%arg0: index):
            "gpu.return"(%arg0) : (index) -> ()
        }) {"sym_name" = "foo", "kernel", "function_type" = (index) -> (index)} : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) : () -> ()

// CHECK: Operation does not verify: Expected void return type for kernel function
