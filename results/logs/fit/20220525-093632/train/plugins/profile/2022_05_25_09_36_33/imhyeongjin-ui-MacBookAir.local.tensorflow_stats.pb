"?2
BHostIDLE"IDLE1     <?@A     <?@a?nN}r???i?nN}r????Unknown
lHostMLCSubgraphOp"MLCSubgraphOp_0_1(1     ??@9     ??@A     ??@I     ??@a???"????i? ?S????Unknown
}HostMLCSubgraphOp""functional_model/MLCSubgraphOp_0_0(1     0?@9     0?@A     0?@I     0?@ay4	??j??i&?B-?????Unknown
^HostGatherV2"GatherV2(1      a@9      a@A      a@I      a@a??[_?>??i,?=p?????Unknown
VHostSum"Sum_2(1      A@9      A@A      A@I      A@a	|?mxw?i$?a????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      >@9      >@A      >@I      >@a51?Qy\t?i?L?S? ???Unknown?
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      8@9      8@A      8@I      8@a?????Ip?i?Ib!???Unknown
dHostDataset"Iterator::Model(1      *@9      *@A      *@I      *@aa??5z?a?i?sA?3???Unknown
}	Host_MLCEagerReadVariableOp"Adam/Cast/ReadVariableOp(1      &@9      &@A      &@I      &@a73??]?i?3[>?A???Unknown
?
HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      &@9      &@A      &@I      &@a73??]?i&?t??P???Unknown
sHostSoftmax"functional_model/dense/Softmax(1      &@9      &@A      &@I      &@a73??]?i²?4?_???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      $@9      $@A      $@I      $@aF???%[?i8x0fm???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@aF???%[?i?=?+?z???Unknown?
ZHostArgMax"ArgMax(1      @9      @A      @I      @a?????IP?i??(????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a?????IP?in?M&C????Unknown
Host_MLCEagerReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @aF???%K?i)??????Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @aF???%K?i???!֘???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      @9      @A      @I      @aF???%K?i?????????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aF???%K?iZ?ei????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?V????E?i?j?׫???Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?V????E?i?S?E????Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a?V????E?i<?????Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?V????E?i?$E!????Unknown?
?Host_MLCEagerReadVariableOp"div_no_nan/ReadVariableOp_1(1      @9      @A      @I      @a?V????E?iH}?????Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a?????I@?i??&??????Unknown
Host_MLCEagerReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a?????I@?i(???????Unknown
~Host_MLCEagerReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a?????I@?i??z??????Unknown
?Host_MLCEagerReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a?????I@?i?$?????Unknown
?HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?????I@?ix?΍?????Unknown
]HostCast"Adam/Cast_1(1       @9       @A       @I       @a?V????5?iéꌢ????Unknown
YHostPow"Adam/Pow(1       @9       @A       @I       @a?V????5?i??Y????Unknown
[ HostPow"
Adam/Pow_1(1       @9       @A       @I       @a?V????5?iY?"?????Unknown
t!HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?V????5?i??>??????Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?V????5?i?zZ?~????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(1       @9       @A       @I       @a?V????5?i:ov?5????Unknown
X$HostEqual"Equal(1       @9       @A       @I       @a?V????5?i?c???????Unknown
?%HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?V????5?i?W???????Unknown
?&HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a?V????5?iLʅZ????Unknown
?'HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1       @9       @A       @I       @a?V????5?if@??????Unknown
?(HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1       @9       @A       @I       @a?V????5?i?4??????Unknown
o)HostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??a?V????%?i?.?$????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a?V????%?i?(?????Unknown
X+HostCast"Cast_2(1      ??9      ??A      ??I      ??a?V????%?i #??????Unknown
X,HostCast"Cast_3(1      ??9      ??A      ??I      ??a?V????%?iE:?6????Unknown
T-HostMul"Mul(1      ??9      ??A      ??I      ??a?V????%?ij??????Unknown
`.HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a?V????%?i?V??????Unknown
?/Host_MLCEagerReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?V????%?i?? I????Unknown
v0HostCast"$sparse_categorical_crossentropy/Cast(1      ??9      ??A      ??I      ??a?V????%?i?r??????Unknown
?1HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?V????%?i?????????Unknown
?2HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?V????%?i?ƿ? ???Unknown
I3HostAssignAddVariableOp"AssignAddVariableOp_3(i?ƿ? ???Unknown
)4HostCast"Cast(i?ƿ? ???Unknown
+5HostCast"Cast_4(i?ƿ? ???Unknown
46HostIdentity"Identity(i?ƿ? ???Unknown?
i7HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i?ƿ? ???Unknown
58HostDivNoNan"div_no_nan_1(i?ƿ? ???Unknown*?1
lHostMLCSubgraphOp"MLCSubgraphOp_0_1(1     ??@9     ??@A     ??@I     ??@aHt??????iHt???????Unknown
}HostMLCSubgraphOp""functional_model/MLCSubgraphOp_0_0(1     0?@9     0?@A     0?@I     0?@a??6????i.?N?I????Unknown
^HostGatherV2"GatherV2(1      a@9      a@A      a@I      a@a?\ ?????i?? ??x???Unknown
VHostSum"Sum_2(1      A@9      A@A      A@I      A@a?????Z??i?!k`?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1      >@9      >@A      >@I      >@a???lȊ?i?Q]???Unknown?
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      8@9      8@A      8@I      8@a??	?#m??ixңƲ???Unknown
dHostDataset"Iterator::Model(1      *@9      *@A      *@I      *@aK?B<6w?i5bX3????Unknown
}Host_MLCEagerReadVariableOp"Adam/Cast/ReadVariableOp(1      &@9      &@A      &@I      &@a{!??s?ix??3{???Unknown
?	HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      &@9      &@A      &@I      &@a{!??s?i???J?/???Unknown
s
HostSoftmax"functional_model/dense/Softmax(1      &@9      &@A      &@I      &@a{!??s?i??aW???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      $@9      $@A      $@I      $@a?2G??q?iR|?H?z???Unknown
iHostWriteSummary"WriteSummary(1      $@9      $@A      $@I      $@a?2G??q?i??/w????Unknown?
ZHostArgMax"ArgMax(1      @9      @A      @I      @a??	?#me?i??S?????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??	?#me?i???vQ????Unknown
Host_MLCEagerReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a?2G??a?i?'=j,????Unknown
[HostAddV2"Adam/add(1      @9      @A      @I      @a?2G??a?i,Z?]????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      @9      @A      @I      @a?2G??a?i֌?P?????Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a?2G??a?i??D????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?v?>??\?i;????Unknown
?HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a?v?>??\?i?vQ?N-???Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a?v?>??\?i?????;???Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a?v?>??\?il.?N?I???Unknown?
?Host_MLCEagerReadVariableOp"div_no_nan/ReadVariableOp_1(1      @9      @A      @I      @a?v?>??\?i'?/)X???Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a??	?#mU?i?'??b???Unknown
Host_MLCEagerReadVariableOp"Adam/Cast_3/ReadVariableOp(1      @9      @A      @I      @a??	?#mU?i??5?m???Unknown
~Host_MLCEagerReadVariableOp"div_no_nan/ReadVariableOp(1      @9      @A      @I      @a??	?#mU?i??Lx???Unknown
?Host_MLCEagerReadVariableOp"div_no_nan_1/ReadVariableOp(1      @9      @A      @I      @a??	?#mU?iW?Y????Unknown
?HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a??	?#mU?i#"빍???Unknown
]HostCast"Adam/Cast_1(1       @9       @A       @I       @a?v?>??L?i?TLޔ???Unknown
YHostPow"Adam/Pow(1       @9       @A       @I       @a?v?>??L?i?}??????Unknown
[HostPow"
Adam/Pow_1(1       @9       @A       @I       @a?v?>??L?i?+?'????Unknown
t HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a?v?>??L?i??CpK????Unknown
v!HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?v?>??L?iy???o????Unknown
v"HostAssignAddVariableOp"AssignAddVariableOp_2(1       @9       @A       @I       @a?v?>??L?iW5?2?????Unknown
X#HostEqual"Equal(1       @9       @A       @I       @a?v?>??L?i5?2??????Unknown
?$HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1       @9       @A       @I       @a?v?>??L?i????????Unknown
?%HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a?v?>??L?i?>?V????Unknown
?&HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1       @9       @A       @I       @a?v?>??L?i??!?%????Unknown
?'HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1       @9       @A       @I       @a?v?>??L?i??qJ????Unknown
o(HostReadVariableOp"Adam/ReadVariableOp(1      ??9      ??A      ??I      ??a?v?>??<?i?qJ?????Unknown
v)HostAssignAddVariableOp"AssignAddVariableOp_4(1      ??9      ??A      ??I      ??a?v?>??<?i?H?zn????Unknown
X*HostCast"Cast_2(1      ??9      ??A      ??I      ??a?v?>??<?izi? ????Unknown
X+HostCast"Cast_3(1      ??9      ??A      ??I      ??a?v?>??<?ii?ܒ????Unknown
T,HostMul"Mul(1      ??9      ??A      ??I      ??a?v?>??<?iX͸%????Unknown
`-HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a?v?>??<?iG?`=?????Unknown
?.Host_MLCEagerReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      ??9      ??A      ??I      ??a?v?>??<?i6{nI????Unknown
v/HostCast"$sparse_categorical_crossentropy/Cast(1      ??9      ??A      ??I      ??a?v?>??<?i%R???????Unknown
?0HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?v?>??<?i)X?m????Unknown
?1HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?v?>??<?i     ???Unknown
I2HostAssignAddVariableOp"AssignAddVariableOp_3(i     ???Unknown
)3HostCast"Cast(i     ???Unknown
+4HostCast"Cast_4(i     ???Unknown
45HostIdentity"Identity(i     ???Unknown?
i6HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(i     ???Unknown
57HostDivNoNan"div_no_nan_1(i     ???Unknown2CPU