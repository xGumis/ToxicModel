бў/
щ╣
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
│
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
е
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ъ
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
Т
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
Б
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718У┬-
Е
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Й'2*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	Й'2*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	м*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Г
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Д*$
shared_namegru/gru_cell/kernel
|
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel*
_output_shapes
:	2Д*
dtype0
Ш
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
мД*.
shared_namegru/gru_cell/recurrent_kernel
С
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel* 
_output_shapes
:
мД*
dtype0

gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д*"
shared_namegru/gru_cell/bias
x
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:	Д*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:╚*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:╚*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:╚*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:╚*
dtype0
Г
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	м*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
С
Adam/gru/gru_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Д*+
shared_nameAdam/gru/gru_cell/kernel/m
К
.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/m*
_output_shapes
:	2Д*
dtype0
ж
$Adam/gru/gru_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
мД*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/m
Я
8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/m* 
_output_shapes
:
мД*
dtype0
Н
Adam/gru/gru_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д*)
shared_nameAdam/gru/gru_cell/bias/m
Ж
,Adam/gru/gru_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/m*
_output_shapes
:	Д*
dtype0
Г
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	м*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
С
Adam/gru/gru_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2Д*+
shared_nameAdam/gru/gru_cell/kernel/v
К
.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/kernel/v*
_output_shapes
:	2Д*
dtype0
ж
$Adam/gru/gru_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
мД*5
shared_name&$Adam/gru/gru_cell/recurrent_kernel/v
Я
8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp$Adam/gru/gru_cell/recurrent_kernel/v* 
_output_shapes
:
мД*
dtype0
Н
Adam/gru/gru_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Д*)
shared_nameAdam/gru/gru_cell/bias/v
Ж
,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru/gru_cell/bias/v*
_output_shapes
:	Д*
dtype0

NoOpNoOp
д,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*▀+
value╒+B╥+ B╦+
є
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
b

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
Ъ
 iter

!beta_1

"beta_2
	#decay
$learning_ratem]m^%m_&m`'mavbvc%vd&ve'vf
*
0
%1
&2
'3
4
5
 
#
%0
&1
'2
3
4
н

(layers
)layer_metrics
	variables
*layer_regularization_losses
+metrics
regularization_losses
,non_trainable_variables
trainable_variables
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
н

-layers
.layer_metrics
	variables
/layer_regularization_losses
0metrics
regularization_losses
1non_trainable_variables
trainable_variables
 
 
 
н

2layers
3layer_metrics
	variables
4layer_regularization_losses
5metrics
regularization_losses
6non_trainable_variables
trainable_variables
~

%kernel
&recurrent_kernel
'bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
 

%0
&1
'2
 

%0
&1
'2
╣

;layers
<layer_metrics
	variables
=layer_regularization_losses
>metrics
regularization_losses

?states
@non_trainable_variables
trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н

Alayers
Blayer_metrics
	variables
Clayer_regularization_losses
Dmetrics
regularization_losses
Enon_trainable_variables
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEgru/gru_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEgru/gru_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 
 

F0
G1
H2

0
 
 
 
 

0
 
 
 
 
 

%0
&1
'2
 

%0
&1
'2
н

Ilayers
Jlayer_metrics
7	variables
Klayer_regularization_losses
Lmetrics
8regularization_losses
Mnon_trainable_variables
9trainable_variables

0
 
 
 
 
 
 
 
 
 
 
4
	Ntotal
	Ocount
P	variables
Q	keras_api
D
	Rtotal
	Scount
T
_fn_kwargs
U	variables
V	keras_api
p
Wtrue_positives
Xtrue_negatives
Yfalse_positives
Zfalse_negatives
[	variables
\	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

P	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

U	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
Y2
Z3

[	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/gru/gru_cell/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/gru/gru_cell/recurrent_kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/gru/gru_cell/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Д
serving_default_embedding_inputPlaceholder*(
_output_shapes
:         °
*
dtype0*
shape:         °

└
StatefulPartitionedCallStatefulPartitionedCallserving_default_embedding_inputembedding/embeddingsgru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biasdense/kernel
dense/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_636657
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
е
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/m/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/m/Read/ReadVariableOp,Adam/gru/gru_cell/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp.Adam/gru/gru_cell/kernel/v/Read/ReadVariableOp8Adam/gru/gru_cell/recurrent_kernel/v/Read/ReadVariableOp,Adam/gru/gru_cell/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_639281
▄
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rategru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativesAdam/dense/kernel/mAdam/dense/bias/mAdam/gru/gru_cell/kernel/m$Adam/gru/gru_cell/recurrent_kernel/mAdam/gru/gru_cell/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/gru/gru_cell/kernel/v$Adam/gru/gru_cell/recurrent_kernel/vAdam/gru/gru_cell/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_639378О╟,
┌
Т
+__inference_sequential_layer_call_fn_636691

inputs
unknown:	Й'2
	unknown_0:	2Д
	unknown_1:
мД
	unknown_2:	Д
	unknown_3:	м
	unknown_4:
identityИвStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6365602
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
ї
Ы
+__inference_sequential_layer_call_fn_636592
embedding_input
unknown:	Й'2
	unknown_0:	2Д
	unknown_1:
мД
	unknown_2:	Д
	unknown_3:	м
	unknown_4:
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6365602
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:         °

)
_user_specified_nameembedding_input
╡
k
2__inference_spatial_dropout1d_layer_call_fn_637523

inputs
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6340212
StatefulPartitionedCallд
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
В
╛
?__inference_gru_layer_call_and_return_conditional_losses_634842

inputs/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2м
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:         м:                  м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6346272
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs
╪D
г
__inference_standard_gru_635813

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_635724*
condR
while_cond_635723*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeъ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°
         м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permз
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_1:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_64f0773c-52de-4eb1-b73e-38eb72b83b2d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪1
т
while_body_636168
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
╘
╢
$__inference_gru_layer_call_fn_637631

inputs
unknown:	2Д
	unknown_0:
мД
	unknown_1:	Д
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6364722
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         °
2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
╪о
ц

:__inference___backward_gpu_gru_with_fallback_634704_634840
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0Е
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:                  м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides▄
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:                  м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationщ
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:                  м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeХ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:                  м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ю
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*V
_output_shapesD
B:                  2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/ReshapeЗ
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*н
_input_shapesЫ
Ш:         м:                  м:         м: :                  м::         м: ::                  2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_1a446bac-57c8-4439-b9af-23640c91187b*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_634839*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:;7
5
_output_shapes#
!:                  м:.*
(
_output_shapes
:         м:

_output_shapes
: :;7
5
_output_shapes#
!:                  м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :                  2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▓	
▐
while_cond_638466
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_638466___redundant_placeholder04
0while_while_cond_638466___redundant_placeholder14
0while_while_cond_638466___redundant_placeholder24
0while_while_cond_638466___redundant_placeholder34
0while_while_cond_638466___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╪1
т
while_body_638467
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
ю;
п
(__inference_gpu_gru_with_fallback_634703

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cс
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_1a446bac-57c8-4439-b9af-23640c91187b*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
п

є
A__inference_dense_layer_call_and_return_conditional_losses_639171

inputs1
matmul_readvariableop_resource:	м-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         м: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
╜;
п
(__inference_gpu_gru_with_fallback_636939

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c┘
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_be090e49-95cf-4abf-9873-89ea5ade3ded*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
▓	
▐
while_cond_637706
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_637706___redundant_placeholder04
0while_while_cond_637706___redundant_placeholder14
0while_while_cond_637706___redundant_placeholder24
0while_while_cond_637706___redundant_placeholder34
0while_while_cond_637706___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
юн
ц

:__inference___backward_gpu_gru_with_fallback_637351_637487
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0}
gradients/grad_ys_1Identityplaceholder_1*
T0*-
_output_shapes
:         °
м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides╘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*-
_output_shapes
:°
         м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationс
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*-
_output_shapes
:°
         м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeН
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*-
_output_shapes
:°
         м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::°
         2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         °
22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/Reshape
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:         °
22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:         м:         °
м:         м: :°
         м::         м: ::°
         2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_fff55791-3bb8-4c20-980a-019a1399ec92*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_637486*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:3/
-
_output_shapes
:         °
м:.*
(
_output_shapes
:         м:

_output_shapes
: :3/
-
_output_shapes
:°
         м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:°
         2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╜;
п
(__inference_gpu_gru_with_fallback_638632

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c┘
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_168a2093-5be7-4fe5-a429-6704cae58d47*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╙
l
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637560

inputs
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/ConstЙ
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'                           2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1═
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape╨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :                  *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y╦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  2
dropout/GreaterEqualМ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  2
dropout/CastР
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'                           2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╠L
┘
F__inference_sequential_layer_call_and_return_conditional_losses_637496

inputs4
!embedding_embedding_lookup_637089:	Й'23
 gru_read_readvariableop_resource:	2Д6
"gru_read_1_readvariableop_resource:
мД5
"gru_read_2_readvariableop_resource:	Д7
$dense_matmul_readvariableop_resource:	м3
%dense_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвembedding/embedding_lookupвgru/Read/ReadVariableOpвgru/Read_1/ReadVariableOpвgru/Read_2/ReadVariableOpr
embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:         °
2
embedding/Cast░
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_637089embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/637089*,
_output_shapes
:         °
2*
dtype02
embedding/embedding_lookupЦ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/637089*,
_output_shapes
:         °
22%
#embedding/embedding_lookup/Identity┐
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:         °
22'
%embedding/embedding_lookup/Identity_1Р
spatial_dropout1d/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
spatial_dropout1d/ShapeШ
%spatial_dropout1d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%spatial_dropout1d/strided_slice/stackЬ
'spatial_dropout1d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout1d/strided_slice/stack_1Ь
'spatial_dropout1d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout1d/strided_slice/stack_2╬
spatial_dropout1d/strided_sliceStridedSlice spatial_dropout1d/Shape:output:0.spatial_dropout1d/strided_slice/stack:output:00spatial_dropout1d/strided_slice/stack_1:output:00spatial_dropout1d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
spatial_dropout1d/strided_sliceЬ
'spatial_dropout1d/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'spatial_dropout1d/strided_slice_1/stackа
)spatial_dropout1d/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout1d/strided_slice_1/stack_1а
)spatial_dropout1d/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)spatial_dropout1d/strided_slice_1/stack_2╪
!spatial_dropout1d/strided_slice_1StridedSlice spatial_dropout1d/Shape:output:00spatial_dropout1d/strided_slice_1/stack:output:02spatial_dropout1d/strided_slice_1/stack_1:output:02spatial_dropout1d/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!spatial_dropout1d/strided_slice_1З
spatial_dropout1d/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2!
spatial_dropout1d/dropout/Const╓
spatial_dropout1d/dropout/MulMul.embedding/embedding_lookup/Identity_1:output:0(spatial_dropout1d/dropout/Const:output:0*
T0*,
_output_shapes
:         °
22
spatial_dropout1d/dropout/Mulж
0spatial_dropout1d/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0spatial_dropout1d/dropout/random_uniform/shape/1з
.spatial_dropout1d/dropout/random_uniform/shapePack(spatial_dropout1d/strided_slice:output:09spatial_dropout1d/dropout/random_uniform/shape/1:output:0*spatial_dropout1d/strided_slice_1:output:0*
N*
T0*
_output_shapes
:20
.spatial_dropout1d/dropout/random_uniform/shapeЖ
6spatial_dropout1d/dropout/random_uniform/RandomUniformRandomUniform7spatial_dropout1d/dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :                  *
dtype028
6spatial_dropout1d/dropout/random_uniform/RandomUniformЩ
(spatial_dropout1d/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2*
(spatial_dropout1d/dropout/GreaterEqual/yУ
&spatial_dropout1d/dropout/GreaterEqualGreaterEqual?spatial_dropout1d/dropout/random_uniform/RandomUniform:output:01spatial_dropout1d/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  2(
&spatial_dropout1d/dropout/GreaterEqual┬
spatial_dropout1d/dropout/CastCast*spatial_dropout1d/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  2 
spatial_dropout1d/dropout/Cast╟
spatial_dropout1d/dropout/Mul_1Mul!spatial_dropout1d/dropout/Mul:z:0"spatial_dropout1d/dropout/Cast:y:0*
T0*,
_output_shapes
:         °
22!
spatial_dropout1d/dropout/Mul_1i
	gru/ShapeShape#spatial_dropout1d/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stackА
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1А
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2·
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slicee
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessk
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
gru/zeros/packed/1У
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/ConstЖ
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:         м2
	gru/zerosФ
gru/Read/ReadVariableOpReadVariableOp gru_read_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
gru/Read/ReadVariableOps
gru/IdentityIdentitygru/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2
gru/IdentityЫ
gru/Read_1/ReadVariableOpReadVariableOp"gru_read_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
gru/Read_1/ReadVariableOpz
gru/Identity_1Identity!gru/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2
gru/Identity_1Ъ
gru/Read_2/ReadVariableOpReadVariableOp"gru_read_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
gru/Read_2/ReadVariableOpy
gru/Identity_2Identity!gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2
gru/Identity_2┘
gru/PartitionedCallPartitionedCall#spatial_dropout1d/dropout/Mul_1:z:0gru/zeros:output:0gru/Identity:output:0gru/Identity_1:output:0gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         м:         °
м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6372742
gru/PartitionedCallа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
dense/MatMul/ReadVariableOpЫ
dense/MatMulMatMulgru/PartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense/SigmoidС
IdentityIdentitydense/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup^gru/Read/ReadVariableOp^gru/Read_1/ReadVariableOp^gru/Read_2/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup22
gru/Read/ReadVariableOpgru/Read/ReadVariableOp26
gru/Read_1/ReadVariableOpgru/Read_1/ReadVariableOp26
gru/Read_2/ReadVariableOpgru/Read_2/ReadVariableOp:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
К
└
?__inference_gru_layer_call_and_return_conditional_losses_638391
inputs_0/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2о
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:         м:                  м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6381762
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :                  2
"
_user_specified_name
inputs/0
юн
ц

:__inference___backward_gpu_gru_with_fallback_633835_633971
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0}
gradients/grad_ys_1Identityplaceholder_1*
T0*-
_output_shapes
:         °
м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides╘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*-
_output_shapes
:°
         м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationс
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*-
_output_shapes
:°
         м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeН
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*-
_output_shapes
:°
         м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::°
         2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         °
22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/Reshape
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:         °
22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:         м:         °
м:         м: :°
         м::         м: ::°
         2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_e5822776-aec5-4613-8e0d-7f78e2ed87f7*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_633970*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:3/
-
_output_shapes
:         °
м:.*
(
_output_shapes
:         м:

_output_shapes
: :3/
-
_output_shapes
:°
         м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:°
         2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▀E
╝
&__forward_gpu_gru_with_fallback_638388

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cх
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_0b30aa8a-c0f3-4058-bfe6-5f09c88b5607*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_638253_638389*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
к
N
2__inference_spatial_dropout1d_layer_call_fn_637518

inputs
identityф
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'                           * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6339892
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
Ы

г
E__inference_embedding_layer_call_and_return_conditional_losses_635639

inputs*
embedding_lookup_635633:	Й'2
identityИвembedding_lookup^
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:         °
2
Cast■
embedding_lookupResourceGatherembedding_lookup_635633Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/635633*,
_output_shapes
:         °
2*
dtype02
embedding_lookupю
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/635633*,
_output_shapes
:         °
22
embedding_lookup/Identityб
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:         °
22
embedding_lookup/Identity_1Р
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:         °
22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         °
: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
ю;
п
(__inference_gpu_gru_with_fallback_637872

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cс
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_a170d7ac-fc4b-4178-8ebc-14307d0762c8*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
▓	
▐
while_cond_638846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_638846___redundant_placeholder04
0while_while_cond_638846___redundant_placeholder14
0while_while_cond_638846___redundant_placeholder24
0while_while_cond_638846___redundant_placeholder34
0while_while_cond_638846___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╖E
╝
&__forward_gpu_gru_with_fallback_636469

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c▌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_3c66f171-ba87-411e-bf85-7866cfe9dee9*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_636334_636470*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪1
т
while_body_634136
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
╪1
т
while_body_637707
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
╖E
╝
&__forward_gpu_gru_with_fallback_636025

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c▌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_64f0773c-52de-4eb1-b73e-38eb72b83b2d*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_635890_636026*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╖E
╝
&__forward_gpu_gru_with_fallback_637075

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c▌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_be090e49-95cf-4abf-9873-89ea5ade3ded*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_636940_637076*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╥
k
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637538

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'                           2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'                           2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
В
╛
?__inference_gru_layer_call_and_return_conditional_losses_634440

inputs/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2м
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:         м:                  м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6342252
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs
╪D
г
__inference_standard_gru_638936

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_638847*
condR
while_cond_638846*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeъ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°
         м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permз
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_1:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_ef76c99a-d847-48fc-8cbe-ced8bdb3690b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
ъ
╕
$__inference_gru_layer_call_fn_637609
inputs_0
unknown:	2Д
	unknown_0:
мД
	unknown_1:	Д
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6348422
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  2
"
_user_specified_name
inputs/0
╘
╢
$__inference_gru_layer_call_fn_637620

inputs
unknown:	2Д
	unknown_0:
мД
	unknown_1:	Д
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6360282
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         °
2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
юн
ц

:__inference___backward_gpu_gru_with_fallback_636940_637076
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0}
gradients/grad_ys_1Identityplaceholder_1*
T0*-
_output_shapes
:         °
м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides╘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*-
_output_shapes
:°
         м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationс
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*-
_output_shapes
:°
         м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeН
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*-
_output_shapes
:°
         м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::°
         2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         °
22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/Reshape
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:         °
22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:         м:         °
м:         м: :°
         м::         м: ::°
         2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_be090e49-95cf-4abf-9873-89ea5ade3ded*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_637075*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:3/
-
_output_shapes
:         °
м:.*
(
_output_shapes
:         м:

_output_shapes
: :3/
-
_output_shapes
:°
         м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:°
         2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
юн
ц

:__inference___backward_gpu_gru_with_fallback_639013_639149
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0}
gradients/grad_ys_1Identityplaceholder_1*
T0*-
_output_shapes
:         °
м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides╘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*-
_output_shapes
:°
         м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationс
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*-
_output_shapes
:°
         м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeН
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*-
_output_shapes
:°
         м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::°
         2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         °
22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/Reshape
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:         °
22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:         м:         °
м:         м: :°
         м::         м: ::°
         2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_ef76c99a-d847-48fc-8cbe-ced8bdb3690b*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_639148*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:3/
-
_output_shapes
:         °
м:.*
(
_output_shapes
:         м:

_output_shapes
: :3/
-
_output_shapes
:°
         м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:°
         2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╙
l
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_634021

inputs
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/ConstЙ
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'                           2
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1═
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape╨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :                  *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y╦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  2
dropout/GreaterEqualМ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  2
dropout/CastР
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'                           2
dropout/Mul_1{
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'                           2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
╪1
т
while_body_634538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
╖E
╝
&__forward_gpu_gru_with_fallback_638768

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c▌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_168a2093-5be7-4fe5-a429-6704cae58d47*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_638633_638769*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╜;
п
(__inference_gpu_gru_with_fallback_639012

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c┘
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_ef76c99a-d847-48fc-8cbe-ced8bdb3690b*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪1
т
while_body_633669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
юн
ц

:__inference___backward_gpu_gru_with_fallback_635890_636026
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0}
gradients/grad_ys_1Identityplaceholder_1*
T0*-
_output_shapes
:         °
м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides╘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*-
_output_shapes
:°
         м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationс
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*-
_output_shapes
:°
         м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeН
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*-
_output_shapes
:°
         м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::°
         2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         °
22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/Reshape
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:         °
22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:         м:         °
м:         м: :°
         м::         м: ::°
         2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_64f0773c-52de-4eb1-b73e-38eb72b83b2d*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_636025*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:3/
-
_output_shapes
:         °
м:.*
(
_output_shapes
:         м:

_output_shapes
: :3/
-
_output_shapes
:°
         м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:°
         2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ЙE
г
__inference_standard_gru_634225

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_634136*
condR
while_cond_634135*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_479bef79-9245-45ba-ae41-062f13ad3e80*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪D
г
__inference_standard_gru_636257

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_636168*
condR
while_cond_636167*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeъ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°
         м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permз
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_1:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_3c66f171-ba87-411e-bf85-7866cfe9dee9*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
ъ
╛
?__inference_gru_layer_call_and_return_conditional_losses_636028

inputs/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2д
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         м:         °
м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6358132
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         °
2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
╜;
п
(__inference_gpu_gru_with_fallback_636333

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c┘
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_3c66f171-ba87-411e-bf85-7866cfe9dee9*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪1
т
while_body_635724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
╪1
т
while_body_638087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
▓	
▐
while_cond_635723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_635723___redundant_placeholder04
0while_while_cond_635723___redundant_placeholder14
0while_while_cond_635723___redundant_placeholder24
0while_while_cond_635723___redundant_placeholder34
0while_while_cond_635723___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
┌
Т
+__inference_sequential_layer_call_fn_636674

inputs
unknown:	Й'2
	unknown_0:	2Д
	unknown_1:
мД
	unknown_2:	Д
	unknown_3:	м
	unknown_4:
identityИвStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6360542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
Д

*__inference_embedding_layer_call_fn_637503

inputs
unknown:	Й'2
identityИвStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_6356392
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         °
22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         °
: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
╜;
п
(__inference_gpu_gru_with_fallback_633834

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c┘
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_e5822776-aec5-4613-8e0d-7f78e2ed87f7*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
юн
ц

:__inference___backward_gpu_gru_with_fallback_638633_638769
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0}
gradients/grad_ys_1Identityplaceholder_1*
T0*-
_output_shapes
:         °
м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides╘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*-
_output_shapes
:°
         м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationс
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*-
_output_shapes
:°
         м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeН
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*-
_output_shapes
:°
         м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::°
         2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         °
22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/Reshape
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:         °
22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:         м:         °
м:         м: :°
         м::         м: ::°
         2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_168a2093-5be7-4fe5-a429-6704cae58d47*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_638768*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:3/
-
_output_shapes
:         °
м:.*
(
_output_shapes
:         м:

_output_shapes
: :3/
-
_output_shapes
:°
         м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:°
         2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ЙE
г
__inference_standard_gru_638176

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_638087*
condR
while_cond_638086*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_0b30aa8a-c0f3-4058-bfe6-5f09c88b5607*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪D
г
__inference_standard_gru_638556

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_638467*
condR
while_cond_638466*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeъ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°
         м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permз
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_1:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_168a2093-5be7-4fe5-a429-6704cae58d47*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
▀E
╝
&__forward_gpu_gru_with_fallback_634839

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cх
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_1a446bac-57c8-4439-b9af-23640c91187b*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_634704_634840*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╜;
п
(__inference_gpu_gru_with_fallback_637350

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c┘
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_fff55791-3bb8-4c20-980a-019a1399ec92*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
№
l
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637587

inputs
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         °
22
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1═
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape╨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :                  *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y╦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  2
dropout/GreaterEqualМ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         °
22
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         °
22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         °
2:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
ЙE
г
__inference_standard_gru_634627

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_634538*
condR
while_cond_634537*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_1a446bac-57c8-4439-b9af-23640c91187b*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪D
г
__inference_standard_gru_637274

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_637185*
condR
while_cond_637184*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeъ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°
         м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permз
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_1:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_fff55791-3bb8-4c20-980a-019a1399ec92*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
п

є
A__inference_dense_layer_call_and_return_conditional_losses_636047

inputs1
matmul_readvariableop_resource:	м-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         м: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
ё
k
2__inference_spatial_dropout1d_layer_call_fn_637533

inputs
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6365102
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         °
22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         °
222
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
ъ
╛
?__inference_gru_layer_call_and_return_conditional_losses_638771

inputs/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2д
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         м:         °
м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6385562
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         °
2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
▓	
▐
while_cond_637184
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_637184___redundant_placeholder04
0while_while_cond_637184___redundant_placeholder14
0while_while_cond_637184___redundant_placeholder24
0while_while_cond_637184___redundant_placeholder34
0while_while_cond_637184___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
╪о
ц

:__inference___backward_gpu_gru_with_fallback_638253_638389
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0Е
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:                  м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides▄
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:                  м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationщ
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:                  м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeХ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:                  м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ю
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*V
_output_shapesD
B:                  2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/ReshapeЗ
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*н
_input_shapesЫ
Ш:         м:                  м:         м: :                  м::         м: ::                  2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_0b30aa8a-c0f3-4058-bfe6-5f09c88b5607*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_638388*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:;7
5
_output_shapes#
!:                  м:.*
(
_output_shapes
:         м:

_output_shapes
: :;7
5
_output_shapes#
!:                  м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :                  2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ЙE
г
__inference_standard_gru_637796

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_637707*
condR
while_cond_637706*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_1:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_a170d7ac-fc4b-4178-8ebc-14307d0762c8*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
Ы
Ф
&__inference_dense_layer_call_fn_639160

inputs
unknown:	м
	unknown_0:
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6360472
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         м: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
ъ
╕
$__inference_gru_layer_call_fn_637598
inputs_0
unknown:	2Д
	unknown_0:
мД
	unknown_1:	Д
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6344402
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  2
"
_user_specified_name
inputs/0
╪о
ц

:__inference___backward_gpu_gru_with_fallback_634302_634438
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0Е
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:                  м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides▄
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:                  м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationщ
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:                  м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeХ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:                  м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ю
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*V
_output_shapesD
B:                  2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/ReshapeЗ
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*н
_input_shapesЫ
Ш:         м:                  м:         м: :                  м::         м: ::                  2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_479bef79-9245-45ba-ae41-062f13ad3e80*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_634437*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:;7
5
_output_shapes#
!:                  м:.*
(
_output_shapes
:         м:

_output_shapes
: :;7
5
_output_shapes#
!:                  м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :                  2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╖E
╝
&__forward_gpu_gru_with_fallback_633970

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c▌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_e5822776-aec5-4613-8e0d-7f78e2ed87f7*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_633835_633971*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪1
т
while_body_638847
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
ю;
п
(__inference_gpu_gru_with_fallback_634301

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cс
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_479bef79-9245-45ba-ae41-062f13ad3e80*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪D
г
__inference_standard_gru_633758

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_633669*
condR
while_cond_633668*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeъ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°
         м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permз
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_1:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_e5822776-aec5-4613-8e0d-7f78e2ed87f7*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
▓	
▐
while_cond_636773
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_636773___redundant_placeholder04
0while_while_cond_636773___redundant_placeholder14
0while_while_cond_636773___redundant_placeholder24
0while_while_cond_636773___redundant_placeholder34
0while_while_cond_636773___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
л|
╙
"__inference__traced_restore_639378
file_prefix8
%assignvariableop_embedding_embeddings:	Й'22
assignvariableop_1_dense_kernel:	м+
assignvariableop_2_dense_bias:&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: 9
&assignvariableop_8_gru_gru_cell_kernel:	2ДD
0assignvariableop_9_gru_gru_cell_recurrent_kernel:
мД8
%assignvariableop_10_gru_gru_cell_bias:	Д#
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: 1
"assignvariableop_15_true_positives:	╚1
"assignvariableop_16_true_negatives:	╚2
#assignvariableop_17_false_positives:	╚2
#assignvariableop_18_false_negatives:	╚:
'assignvariableop_19_adam_dense_kernel_m:	м3
%assignvariableop_20_adam_dense_bias_m:A
.assignvariableop_21_adam_gru_gru_cell_kernel_m:	2ДL
8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_m:
мД?
,assignvariableop_23_adam_gru_gru_cell_bias_m:	Д:
'assignvariableop_24_adam_dense_kernel_v:	м3
%assignvariableop_25_adam_dense_bias_v:A
.assignvariableop_26_adam_gru_gru_cell_kernel_v:	2ДL
8assignvariableop_27_adam_gru_gru_cell_recurrent_kernel_v:
мД?
,assignvariableop_28_adam_gru_gru_cell_bias_v:	Д
identity_30ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ш
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ї
valueъBчB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╩
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices┬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*М
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityд
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2в
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3б
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4г
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5г
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6в
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7к
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8л
AssignVariableOp_8AssignVariableOp&assignvariableop_8_gru_gru_cell_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╡
AssignVariableOp_9AssignVariableOp0assignvariableop_9_gru_gru_cell_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10н
AssignVariableOp_10AssignVariableOp%assignvariableop_10_gru_gru_cell_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11б
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12б
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13г
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14г
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15к
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOp"assignvariableop_16_true_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17л
AssignVariableOp_17AssignVariableOp#assignvariableop_17_false_positivesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18л
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19п
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20н
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╢
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_gru_gru_cell_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22└
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adam_gru_gru_cell_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┤
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_gru_gru_cell_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24п
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25н
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_dense_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╢
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_gru_gru_cell_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27└
AssignVariableOp_27AssignVariableOp8assignvariableop_27_adam_gru_gru_cell_recurrent_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28┤
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_gru_gru_cell_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp▄
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29╧
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
И<
┴
!__inference__wrapped_model_633980
embedding_input?
,sequential_embedding_embedding_lookup_633590:	Й'2>
+sequential_gru_read_readvariableop_resource:	2ДA
-sequential_gru_read_1_readvariableop_resource:
мД@
-sequential_gru_read_2_readvariableop_resource:	ДB
/sequential_dense_matmul_readvariableop_resource:	м>
0sequential_dense_biasadd_readvariableop_resource:
identityИв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв%sequential/embedding/embedding_lookupв"sequential/gru/Read/ReadVariableOpв$sequential/gru/Read_1/ReadVariableOpв$sequential/gru/Read_2/ReadVariableOpС
sequential/embedding/CastCastembedding_input*

DstT0*

SrcT0*(
_output_shapes
:         °
2
sequential/embedding/Castч
%sequential/embedding/embedding_lookupResourceGather,sequential_embedding_embedding_lookup_633590sequential/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*?
_class5
31loc:@sequential/embedding/embedding_lookup/633590*,
_output_shapes
:         °
2*
dtype02'
%sequential/embedding/embedding_lookup┬
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@sequential/embedding/embedding_lookup/633590*,
_output_shapes
:         °
220
.sequential/embedding/embedding_lookup/Identityр
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:         °
222
0sequential/embedding/embedding_lookup/Identity_1╠
%sequential/spatial_dropout1d/IdentityIdentity9sequential/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         °
22'
%sequential/spatial_dropout1d/IdentityК
sequential/gru/ShapeShape.sequential/spatial_dropout1d/Identity:output:0*
T0*
_output_shapes
:2
sequential/gru/ShapeТ
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/gru/strided_slice/stackЦ
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_1Ц
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_2╝
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/gru/strided_slice{
sequential/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
sequential/gru/zeros/mul/yи
sequential/gru/zeros/mulMul%sequential/gru/strided_slice:output:0#sequential/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/mul}
sequential/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
sequential/gru/zeros/Less/yг
sequential/gru/zeros/LessLesssequential/gru/zeros/mul:z:0$sequential/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/LessБ
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
sequential/gru/zeros/packed/1┐
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/gru/zeros/packed}
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/zeros/Const▓
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*(
_output_shapes
:         м2
sequential/gru/zeros╡
"sequential/gru/Read/ReadVariableOpReadVariableOp+sequential_gru_read_readvariableop_resource*
_output_shapes
:	2Д*
dtype02$
"sequential/gru/Read/ReadVariableOpФ
sequential/gru/IdentityIdentity*sequential/gru/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2
sequential/gru/Identity╝
$sequential/gru/Read_1/ReadVariableOpReadVariableOp-sequential_gru_read_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02&
$sequential/gru/Read_1/ReadVariableOpЫ
sequential/gru/Identity_1Identity,sequential/gru/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2
sequential/gru/Identity_1╗
$sequential/gru/Read_2/ReadVariableOpReadVariableOp-sequential_gru_read_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02&
$sequential/gru/Read_2/ReadVariableOpЪ
sequential/gru/Identity_2Identity,sequential/gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2
sequential/gru/Identity_2ж
sequential/gru/PartitionedCallPartitionedCall.sequential/spatial_dropout1d/Identity:output:0sequential/gru/zeros:output:0 sequential/gru/Identity:output:0"sequential/gru/Identity_1:output:0"sequential/gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         м:         °
м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6337582 
sequential/gru/PartitionedCall┴
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype02(
&sequential/dense/MatMul/ReadVariableOp╟
sequential/dense/MatMulMatMul'sequential/gru/PartitionedCall:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense/MatMul┐
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp┼
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense/BiasAddФ
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential/dense/Sigmoid▐
IdentityIdentitysequential/dense/Sigmoid:y:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup#^sequential/gru/Read/ReadVariableOp%^sequential/gru/Read_1/ReadVariableOp%^sequential/gru/Read_2/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2H
"sequential/gru/Read/ReadVariableOp"sequential/gru/Read/ReadVariableOp2L
$sequential/gru/Read_1/ReadVariableOp$sequential/gru/Read_1/ReadVariableOp2L
$sequential/gru/Read_2/ReadVariableOp$sequential/gru/Read_2/ReadVariableOp:Y U
(
_output_shapes
:         °

)
_user_specified_nameembedding_input
▓	
▐
while_cond_633668
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_633668___redundant_placeholder04
0while_while_cond_633668___redundant_placeholder14
0while_while_cond_633668___redundant_placeholder24
0while_while_cond_633668___redundant_placeholder34
0while_while_cond_633668___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
ъ
╛
?__inference_gru_layer_call_and_return_conditional_losses_639151

inputs/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2д
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         м:         °
м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6389362
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         °
2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
╥
k
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_633989

inputs

identity_1p
IdentityIdentityinputs*
T0*=
_output_shapes+
):'                           2

Identity

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'                           2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'                           :e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs
ю;
п
(__inference_gpu_gru_with_fallback_638252

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cс
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_0b30aa8a-c0f3-4058-bfe6-5f09c88b5607*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╖E
╝
&__forward_gpu_gru_with_fallback_637486

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c▌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_fff55791-3bb8-4c20-980a-019a1399ec92*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_637351_637487*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
¤0
┘
F__inference_sequential_layer_call_and_return_conditional_losses_637085

inputs4
!embedding_embedding_lookup_636695:	Й'23
 gru_read_readvariableop_resource:	2Д6
"gru_read_1_readvariableop_resource:
мД5
"gru_read_2_readvariableop_resource:	Д7
$dense_matmul_readvariableop_resource:	м3
%dense_biasadd_readvariableop_resource:
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвembedding/embedding_lookupвgru/Read/ReadVariableOpвgru/Read_1/ReadVariableOpвgru/Read_2/ReadVariableOpr
embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:         °
2
embedding/Cast░
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_636695embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/636695*,
_output_shapes
:         °
2*
dtype02
embedding/embedding_lookupЦ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/636695*,
_output_shapes
:         °
22%
#embedding/embedding_lookup/Identity┐
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:         °
22'
%embedding/embedding_lookup/Identity_1л
spatial_dropout1d/IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:         °
22
spatial_dropout1d/Identityi
	gru/ShapeShape#spatial_dropout1d/Identity:output:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stackА
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1А
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2·
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slicee
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessk
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
gru/zeros/packed/1У
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/ConstЖ
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*(
_output_shapes
:         м2
	gru/zerosФ
gru/Read/ReadVariableOpReadVariableOp gru_read_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
gru/Read/ReadVariableOps
gru/IdentityIdentitygru/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2
gru/IdentityЫ
gru/Read_1/ReadVariableOpReadVariableOp"gru_read_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
gru/Read_1/ReadVariableOpz
gru/Identity_1Identity!gru/Read_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2
gru/Identity_1Ъ
gru/Read_2/ReadVariableOpReadVariableOp"gru_read_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
gru/Read_2/ReadVariableOpy
gru/Identity_2Identity!gru/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2
gru/Identity_2┘
gru/PartitionedCallPartitionedCall#spatial_dropout1d/Identity:output:0gru/zeros:output:0gru/Identity:output:0gru/Identity_1:output:0gru/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         м:         °
м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6368632
gru/PartitionedCallа
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
dense/MatMul/ReadVariableOpЫ
dense/MatMulMatMulgru/PartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense/SigmoidС
IdentityIdentitydense/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup^gru/Read/ReadVariableOp^gru/Read_1/ReadVariableOp^gru/Read_2/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup22
gru/Read/ReadVariableOpgru/Read/ReadVariableOp26
gru/Read_1/ReadVariableOpgru/Read_1/ReadVariableOp26
gru/Read_2/ReadVariableOpgru/Read_2/ReadVariableOp:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
х
N
2__inference_spatial_dropout1d_layer_call_fn_637528

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6356472
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         °
22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         °
2:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
▓	
▐
while_cond_634135
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_634135___redundant_placeholder04
0while_while_cond_634135___redundant_placeholder14
0while_while_cond_634135___redundant_placeholder24
0while_while_cond_634135___redundant_placeholder34
0while_while_cond_634135___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
№
l
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_636510

inputs
identityИD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ь
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█╢?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         °
22
dropout/MulВ
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2 
dropout/random_uniform/shape/1═
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
dropout/random_uniform/shape╨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :                  *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y╦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :                  2
dropout/GreaterEqualМ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :                  2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         °
22
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         °
22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         °
2:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
К
└
?__inference_gru_layer_call_and_return_conditional_losses_638011
inputs_0/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2о
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *_
_output_shapesM
K:         м:                  м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6377962
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :                  2
"
_user_specified_name
inputs/0
╪
│
F__inference_sequential_layer_call_and_return_conditional_losses_636560

inputs#
embedding_636543:	Й'2

gru_636547:	2Д

gru_636549:
мД

gru_636551:	Д
dense_636554:	м
dense_636556:
identityИвdense/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвgru/StatefulPartitionedCallв)spatial_dropout1d/StatefulPartitionedCallН
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_636543*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_6356392#
!embedding/StatefulPartitionedCall│
)spatial_dropout1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6365102+
)spatial_dropout1d/StatefulPartitionedCall╣
gru/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout1d/StatefulPartitionedCall:output:0
gru_636547
gru_636549
gru_636551*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6364722
gru/StatefulPartitionedCallж
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_636554dense_636556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6360472
dense/StatefulPartitionedCallИ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*^spatial_dropout1d/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2V
)spatial_dropout1d/StatefulPartitionedCall)spatial_dropout1d/StatefulPartitionedCall:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
▀E
╝
&__forward_gpu_gru_with_fallback_634437

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cх
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_479bef79-9245-45ba-ae41-062f13ad3e80*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_634302_634438*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╔
Ф
$__inference_signature_wrapper_636657
embedding_input
unknown:	Й'2
	unknown_0:	2Д
	unknown_1:
мД
	unknown_2:	Д
	unknown_3:	м
	unknown_4:
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_6339802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:         °

)
_user_specified_nameembedding_input
И
З
F__inference_sequential_layer_call_and_return_conditional_losses_636054

inputs#
embedding_635640:	Й'2

gru_636029:	2Д

gru_636031:
мД

gru_636033:	Д
dense_636048:	м
dense_636050:
identityИвdense/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвgru/StatefulPartitionedCallН
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_635640*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_6356392#
!embedding/StatefulPartitionedCallЫ
!spatial_dropout1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6356472#
!spatial_dropout1d/PartitionedCall▒
gru/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout1d/PartitionedCall:output:0
gru_636029
gru_636031
gru_636033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6360282
gru/StatefulPartitionedCallж
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_636048dense_636050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6360472
dense/StatefulPartitionedCall▄
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
╖E
╝
&__forward_gpu_gru_with_fallback_639148

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c▌
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_ef76c99a-d847-48fc-8cbe-ced8bdb3690b*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_639013_639149*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╜;
п
(__inference_gpu_gru_with_fallback_635889

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Иu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim~

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╒
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes

:Ал2
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_c┘
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*K
_output_shapes9
7:°
         м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permН
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_7:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_64f0773c-52de-4eb1-b73e-38eb72b83b2d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
╪1
т
while_body_637185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
Ы

г
E__inference_embedding_layer_call_and_return_conditional_losses_637513

inputs*
embedding_lookup_637507:	Й'2
identityИвembedding_lookup^
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:         °
2
Cast■
embedding_lookupResourceGatherembedding_lookup_637507Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/637507*,
_output_shapes
:         °
2*
dtype02
embedding_lookupю
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/637507*,
_output_shapes
:         °
22
embedding_lookup/Identityб
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:         °
22
embedding_lookup/Identity_1Р
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*,
_output_shapes
:         °
22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:         °
: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:         °

 
_user_specified_nameinputs
╪1
т
while_body_636774
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         2*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemв
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMulН
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAddp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim╟
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/splitХ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:         Д2
while/MatMul_1Х
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*(
_output_shapes
:         Д2
while/BiasAdd_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim╧
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
while/split_1А
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*(
_output_shapes
:         м2
	while/addk
while/SigmoidSigmoidwhile/add:z:0*
T0*(
_output_shapes
:         м2
while/SigmoidД
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*(
_output_shapes
:         м2
while/add_1q
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*(
_output_shapes
:         м2
while/Sigmoid_1}
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*(
_output_shapes
:         м2
	while/mul{
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*(
_output_shapes
:         м2
while/add_2d

while/TanhTanhwhile/add_2:z:0*
T0*(
_output_shapes
:         м2

while/Tanh|
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*(
_output_shapes
:         м2
while/mul_1_
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
while/sub/xy
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*(
_output_shapes
:         м2
	while/subs
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*(
_output_shapes
:         м2
while/mul_2x
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*(
_output_shapes
:         м2
while/add_3╙
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_4/yo
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: 2
while/add_4`
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_5/yv
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: 2
while/add_5^
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: 2
while/Identity_2Н
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/add_3:z:0*
T0*(
_output_shapes
:         м2
while/Identity_4"4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	2Д:!

_output_shapes	
:Д:&	"
 
_output_shapes
:
мД:!


_output_shapes	
:Д
▓	
▐
while_cond_634537
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_634537___redundant_placeholder04
0while_while_cond_634537___redundant_placeholder14
0while_while_cond_634537___redundant_placeholder24
0while_while_cond_634537___redundant_placeholder34
0while_while_cond_634537___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
▓	
▐
while_cond_638086
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_638086___redundant_placeholder04
0while_while_cond_638086___redundant_placeholder14
0while_while_cond_638086___redundant_placeholder24
0while_while_cond_638086___redundant_placeholder34
0while_while_cond_638086___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
в@
Х
__inference__traced_save_639281
file_prefix3
/savev2_embedding_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_m_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop9
5savev2_adam_gru_gru_cell_kernel_v_read_readvariableopC
?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop7
3savev2_adam_gru_gru_cell_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameт
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ї
valueъBчB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names─
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesМ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop5savev2_adam_gru_gru_cell_kernel_m_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_m_read_readvariableop3savev2_adam_gru_gru_cell_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop5savev2_adam_gru_gru_cell_kernel_v_read_readvariableop?savev2_adam_gru_gru_cell_recurrent_kernel_v_read_readvariableop3savev2_adam_gru_gru_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ы
_input_shapes┘
╓: :	Й'2:	м:: : : : : :	2Д:
мД:	Д: : : : :╚:╚:╚:╚:	м::	2Д:
мД:	Д:	м::	2Д:
мД:	Д: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Й'2:%!

_output_shapes
:	м: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	2Д:&
"
 
_output_shapes
:
мД:%!

_output_shapes
:	Д:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:!

_output_shapes	
:╚:%!

_output_shapes
:	м: 

_output_shapes
::%!

_output_shapes
:	2Д:&"
 
_output_shapes
:
мД:%!

_output_shapes
:	Д:%!

_output_shapes
:	м: 

_output_shapes
::%!

_output_shapes
:	2Д:&"
 
_output_shapes
:
мД:%!

_output_shapes
:	Д:

_output_shapes
: 
О
k
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637565

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         °
22

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         °
22

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         °
2:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
г
Р
F__inference_sequential_layer_call_and_return_conditional_losses_636612
embedding_input#
embedding_636595:	Й'2

gru_636599:	2Д

gru_636601:
мД

gru_636603:	Д
dense_636606:	м
dense_636608:
identityИвdense/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвgru/StatefulPartitionedCallЦ
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_636595*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_6356392#
!embedding/StatefulPartitionedCallЫ
!spatial_dropout1d/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6356472#
!spatial_dropout1d/PartitionedCall▒
gru/StatefulPartitionedCallStatefulPartitionedCall*spatial_dropout1d/PartitionedCall:output:0
gru_636599
gru_636601
gru_636603*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6360282
gru/StatefulPartitionedCallж
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_636606dense_636608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6360472
dense/StatefulPartitionedCall▄
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:Y U
(
_output_shapes
:         °

)
_user_specified_nameembedding_input
юн
ц

:__inference___backward_gpu_gru_with_fallback_636334_636470
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0}
gradients/grad_ys_1Identityplaceholder_1*
T0*-
_output_shapes
:         °
м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides╘
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*-
_output_shapes
:°
         м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationс
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*-
_output_shapes
:°
         м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeН
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*-
_output_shapes
:°
         м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ц
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*N
_output_shapes<
::°
         2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*,
_output_shapes
:         °
22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/Reshape
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*,
_output_shapes
:         °
22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А:         м:         °
м:         м: :°
         м::         м: ::°
         2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_3c66f171-ba87-411e-bf85-7866cfe9dee9*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_636469*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:3/
-
_output_shapes
:         °
м:.*
(
_output_shapes
:         м:

_output_shapes
: :3/
-
_output_shapes
:°
         м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::2	.
,
_output_shapes
:°
         2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
О
k
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_635647

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         °
22

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         °
22

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         °
2:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
╪D
г
__inference_standard_gru_636863

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3Z
unstackUnpackbias*
T0*"
_output_shapes
:Д:Д*	
num2	
unstacku
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:°
         22
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    2   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         2*
shrink_axis_mask2
strided_slice_1o
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:         Д2
MatMult
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         Д2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimп
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2
splitk
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:         Д2

MatMul_1z
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         Д2
	BiasAdd_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim╖
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*P
_output_shapes>
<:         м:         м:         м*
	num_split2	
split_1h
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         м2
addY
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         м2	
Sigmoidl
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         м2
add_1_
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         м2
	Sigmoid_1e
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         м2
mulc
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         м2
add_2R
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         м2
Tanh]
mul_1MulSigmoid:y:0init_h*
T0*(
_output_shapes
:         м2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
sub/xa
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         м2
sub[
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         м2
mul_2`
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         м2
add_3П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterд
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*Y
_output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д* 
_read_only_resource_inputs
 *
bodyR
while_body_636774*
condR
while_cond_636773*X
output_shapesG
E: : : : :         м: : :	2Д:Д:
мД:Д*
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ,  22
0TensorArrayV2Stack/TensorListStack/element_shapeъ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:°
         м*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ы
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permз
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:         °
м2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtimem
IdentityIdentitystrided_slice_2:output:0*
T0*(
_output_shapes
:         м2

Identitym

Identity_1Identitytranspose_1:y:0*
T0*-
_output_shapes
:         °
м2

Identity_1g

Identity_2Identitywhile:output:4*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:         °
2:         м:	2Д:
мД:	Д*<
api_implements*(gru_be090e49-95cf-4abf-9873-89ea5ade3ded*
api_preferred_deviceCPU*
go_backwards( *

time_major( :T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
ъ
╛
?__inference_gru_layer_call_and_return_conditional_losses_636472

inputs/
read_readvariableop_resource:	2Д2
read_1_readvariableop_resource:
мД1
read_2_readvariableop_resource:	Д

identity_3ИвRead/ReadVariableOpвRead_1/ReadVariableOpвRead_2/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :м2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :м2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         м2
zerosИ
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	2Д*
dtype02
Read/ReadVariableOpg
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	2Д2

IdentityП
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource* 
_output_shapes
:
мД*
dtype02
Read_1/ReadVariableOpn

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0* 
_output_shapes
:
мД2

Identity_1О
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes
:	Д*
dtype02
Read_2/ReadVariableOpm

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	Д2

Identity_2д
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         м:         °
м:         м: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference_standard_gru_6362572
PartitionedCall╖

Identity_3IdentityPartitionedCall:output:0^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity_3"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         °
2: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:T P
,
_output_shapes
:         °
2
 
_user_specified_nameinputs
╪о
ц

:__inference___backward_gpu_gru_with_fallback_637873_638009
placeholder
placeholder_1
placeholder_2
placeholder_3/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_7_grad_invertpermutation_transpose_7_perm)
%gradients_squeeze_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4Иv
gradients/grad_ys_0Identityplaceholder*
T0*(
_output_shapes
:         м2
gradients/grad_ys_0Е
gradients/grad_ys_1Identityplaceholder_1*
T0*5
_output_shapes#
!:                  м2
gradients/grad_ys_1x
gradients/grad_ys_2Identityplaceholder_2*
T0*(
_output_shapes
:         м2
gradients/grad_ys_2f
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 2
gradients/grad_ys_3г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shape╜
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
         25
3gradients/strided_slice_grad/StridedSliceGrad/begin░
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 23
1gradients/strided_slice_grad/StridedSliceGrad/end╕
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:27
5gradients/strided_slice_grad/StridedSliceGrad/strides▄
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*5
_output_shapes#
!:                  м*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutationщ
$gradients/transpose_7_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*5
_output_shapes#
!:                  м2&
$gradients/transpose_7_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╟
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*,
_output_shapes
:         м2 
gradients/Squeeze_grad/ReshapeХ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_7_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*5
_output_shapes#
!:                  м2
gradients/AddNy
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: 2
gradients/zeros_likeБ
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like_1Ю
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*V
_output_shapesD
B:                  2:         м: :Ал*
rnn_modegru2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  22$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeы
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*(
_output_shapes
:         м2#
!gradients/ExpandDims_grad/Reshapez
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_1Й
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:Шu2
gradients/concat_grad/Shape_2К
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_3К
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_4К
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Р┐2
gradients/concat_grad/Shape_5Й
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_6Й
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_7Й
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_8Й
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:м2
gradients/concat_grad/Shape_9Л
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_10Л
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/concat_grad/Shape_11╛
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::2$
"gradients/concat_grad/ConcatOffsetЕ
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/SliceЛ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_1Л
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:Шu2
gradients/concat_grad/Slice_2М
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_3М
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_4М
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes

:Р┐2
gradients/concat_grad/Slice_5Л
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_6Л
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_7Л
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_8Л
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes	
:м2
gradients/concat_grad/Slice_9П
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_10П
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes	
:м2 
gradients/concat_grad/Slice_11С
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_1_grad/Shape╚
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_2_grad/Shape╩
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  2   2 
gradients/Reshape_3_grad/Shape╩
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes
:	м22"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB",  ,  2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0* 
_output_shapes
:
мм2"
 gradients/Reshape_6_grad/ReshapeЛ
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_7_grad/Shape╞
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_7_grad/ReshapeЛ
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_8_grad/Shape╞
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_8_grad/ReshapeЛ
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2 
gradients/Reshape_9_grad/Shape╞
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes	
:м2"
 gradients/Reshape_9_grad/ReshapeН
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_10_grad/Shape╔
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_10_grad/ReshapeН
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_11_grad/Shape╩
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_11_grad/ReshapeН
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:м2!
gradients/Reshape_12_grad/Shape╩
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes	
:м2#
!gradients/Reshape_12_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutationр
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutationр
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutationр
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes
:	2м2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutationс
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutationс
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutationс
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0* 
_output_shapes
:
мм2&
$gradients/transpose_6_grad/transposeп
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:И2
gradients/split_2_grad/concatд
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	2Д2
gradients/split_grad/concatн
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0* 
_output_shapes
:
мД2
gradients/split_1_grad/concatН
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   Д  2
gradients/Reshape_grad/Shape─
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes
:	Д2 
gradients/Reshape_grad/ReshapeЗ
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :                  22

IdentityГ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*(
_output_shapes
:         м2

Identity_1t

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	2Д2

Identity_2w

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0* 
_output_shapes
:
мД2

Identity_3w

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes
:	Д2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*н
_input_shapesЫ
Ш:         м:                  м:         м: :                  м::         м: ::                  2:         м: :Ал::         м: ::::::: : : *<
api_implements*(gru_a170d7ac-fc4b-4178-8ebc-14307d0762c8*
api_preferred_deviceGPU*A
forward_function_name(&__forward_gpu_gru_with_fallback_638008*
go_backwards( *

time_major( :. *
(
_output_shapes
:         м:;7
5
_output_shapes#
!:                  м:.*
(
_output_shapes
:         м:

_output_shapes
: :;7
5
_output_shapes#
!:                  м: 

_output_shapes
::2.
,
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :                  2:2
.
,
_output_shapes
:         м:

_output_shapes
: :"

_output_shapes

:Ал: 

_output_shapes
::.*
(
_output_shapes
:         м:

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
є
╝
F__inference_sequential_layer_call_and_return_conditional_losses_636632
embedding_input#
embedding_636615:	Й'2

gru_636619:	2Д

gru_636621:
мД

gru_636623:	Д
dense_636626:	м
dense_636628:
identityИвdense/StatefulPartitionedCallв!embedding/StatefulPartitionedCallвgru/StatefulPartitionedCallв)spatial_dropout1d/StatefulPartitionedCallЦ
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_636615*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_embedding_layer_call_and_return_conditional_losses_6356392#
!embedding/StatefulPartitionedCall│
)spatial_dropout1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         °
2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_6365102+
)spatial_dropout1d/StatefulPartitionedCall╣
gru/StatefulPartitionedCallStatefulPartitionedCall2spatial_dropout1d/StatefulPartitionedCall:output:0
gru_636619
gru_636621
gru_636623*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6364722
gru/StatefulPartitionedCallж
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_636626dense_636628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6360472
dense/StatefulPartitionedCallИ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall^gru/StatefulPartitionedCall*^spatial_dropout1d/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall2V
)spatial_dropout1d/StatefulPartitionedCall)spatial_dropout1d/StatefulPartitionedCall:Y U
(
_output_shapes
:         °

)
_user_specified_nameembedding_input
ї
Ы
+__inference_sequential_layer_call_fn_636069
embedding_input
unknown:	Й'2
	unknown_0:	2Д
	unknown_1:
мД
	unknown_2:	Д
	unknown_3:	м
	unknown_4:
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_6360542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         °
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:         °

)
_user_specified_nameembedding_input
▀E
╝
&__forward_gpu_gru_with_fallback_638008

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
cudnnrnn
transpose_7_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimИu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dimА

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:         м2

ExpandDimsd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimК
splitSplitsplit/split_dim:output:0kernel*
T0*5
_output_shapes#
!:	2м:	2м:	2м*
	num_split2
splith
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dimЭ
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*8
_output_shapes&
$:
мм:
мм:
мм*
	num_split2	
split_1q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
Reshape/shapea
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:И2	
Reshapeh
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_2/split_dimг
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*>
_output_shapes,
*:м:м:м:м:м:м*
	num_split2	
split_2a
ConstConst*
_output_shapes
:*
dtype0*
valueB:
         2
Constu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm|
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes
:	м22
transpose_1h
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_1u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm|
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes
:	м22
transpose_2h
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm|
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes
:	м22
transpose_3h
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:Шu2
	Reshape_3u
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_4i
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_4u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_5i
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_5u
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_6/perm
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0* 
_output_shapes
:
мм2
transpose_6i
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes

:Р┐2
	Reshape_6i
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_7i
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_8i
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes	
:м2
	Reshape_9k

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes	
:м2

Reshape_10k

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes	
:м2

Reshape_11k

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes	
:м2

Reshape_12\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╣
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T02
concati
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    2
CudnnRNN/input_cх
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*S
_output_shapesA
?:                  м:         м: :*
rnn_modegru2

CudnnRNN}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ў
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         м*
shrink_axis_mask2
strided_slicey
transpose_7/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_7/permХ
transpose_7	TransposeCudnnRNN:output:0transpose_7/perm:output:0*
T0*5
_output_shapes#
!:                  м2
transpose_7|
SqueezeSqueezeCudnnRNN:output_h:0*
T0*(
_output_shapes
:         м*
squeeze_dims
 2	
Squeezef
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @2	
runtimek
IdentityIdentitystrided_slice:output:0*
T0*(
_output_shapes
:         м2

Identityu

Identity_1Identitytranspose_7:y:0*
T0*5
_output_shapes#
!:                  м2

Identity_1i

Identity_2IdentitySqueeze:output:0*
T0*(
_output_shapes
:         м2

Identity_2W

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: 2

Identity_3"
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*i
_input_shapesX
V:                  2:         м:	2Д:
мД:	Д*<
api_implements*(gru_a170d7ac-fc4b-4178-8ebc-14307d0762c8*
api_preferred_deviceGPU*V
backward_function_name<:__inference___backward_gpu_gru_with_fallback_637873_638009*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :                  2
 
_user_specified_nameinputs:PL
(
_output_shapes
:         м
 
_user_specified_nameinit_h:GC

_output_shapes
:	2Д
 
_user_specified_namekernel:RN
 
_output_shapes
:
мД
*
_user_specified_namerecurrent_kernel:EA

_output_shapes
:	Д

_user_specified_namebias
▓	
▐
while_cond_636167
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice4
0while_while_cond_636167___redundant_placeholder04
0while_while_cond_636167___redundant_placeholder14
0while_while_cond_636167___redundant_placeholder24
0while_while_cond_636167___redundant_placeholder34
0while_while_cond_636167___redundant_placeholder4
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :         м: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         м:

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╣
serving_defaultе
L
embedding_input9
!serving_default_embedding_input:0         °
9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict:пХ
ЄO
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
regularization_losses
trainable_variables
		keras_api


signatures
g__call__
*h&call_and_return_all_conditional_losses
i_default_save_signature"еM
_tf_keras_sequentialЖM{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "input_dim": 5001, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1400}}, {"class_name": "SpatialDropout1D", "config": {"name": "spatial_dropout1d", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 300, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 6, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1400]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1400]}, "float32", "embedding_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}, "shared_object_id": 0}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "input_dim": 5001, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 1}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1400}, "shared_object_id": 2}, {"class_name": "SpatialDropout1D", "config": {"name": "spatial_dropout1d", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 3}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 300, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 6, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 13}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}, "shared_object_id": 14}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
▄

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"╜
_tf_keras_layerг{"name": "embedding", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1400]}, "dtype": "float32", "input_dim": 5001, "output_dim": 50, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 1}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1400}, "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1400]}}
╜
	variables
regularization_losses
trainable_variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"о
_tf_keras_layerФ{"name": "spatial_dropout1d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "SpatialDropout1D", "config": {"name": "spatial_dropout1d", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 15}}
и
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses" 

_tf_keras_rnn_layerс
{"name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 300, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 8, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 50]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 16}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1400, 50]}}
╧

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"к
_tf_keras_layerР{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 6, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
н
 iter

!beta_1

"beta_2
	#decay
$learning_ratem]m^%m_&m`'mavbvc%vd&ve'vf"
	optimizer
J
0
%1
&2
'3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
C
%0
&1
'2
3
4"
trackable_list_wrapper
╩

(layers
)layer_metrics
	variables
*layer_regularization_losses
+metrics
regularization_losses
,non_trainable_variables
trainable_variables
g__call__
i_default_save_signature
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
,
rserving_default"
signature_map
':%	Й'22embedding/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н

-layers
.layer_metrics
	variables
/layer_regularization_losses
0metrics
regularization_losses
1non_trainable_variables
trainable_variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н

2layers
3layer_metrics
	variables
4layer_regularization_losses
5metrics
regularization_losses
6non_trainable_variables
trainable_variables
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
∙

%kernel
&recurrent_kernel
'bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
s__call__
*t&call_and_return_all_conditional_losses"╛
_tf_keras_layerд{"name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 300, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "shared_object_id": 7}
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
╣

;layers
<layer_metrics
	variables
=layer_regularization_losses
>metrics
regularization_losses

?states
@non_trainable_variables
trainable_variables
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:	м2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н

Alayers
Blayer_metrics
	variables
Clayer_regularization_losses
Dmetrics
regularization_losses
Enon_trainable_variables
trainable_variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
&:$	2Д2gru/gru_cell/kernel
1:/
мД2gru/gru_cell/recurrent_kernel
$:"	Д2gru/gru_cell/bias
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
%0
&1
'2"
trackable_list_wrapper
н

Ilayers
Jlayer_metrics
7	variables
Klayer_regularization_losses
Lmetrics
8regularization_losses
Mnon_trainable_variables
9trainable_variables
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╘
	Ntotal
	Ocount
P	variables
Q	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 18}
Ч
	Rtotal
	Scount
T
_fn_kwargs
U	variables
V	keras_api"╨
_tf_keras_metric╡{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 13}
╟"
Wtrue_positives
Xtrue_negatives
Yfalse_positives
Zfalse_negatives
[	variables
\	keras_api"╘!
_tf_keras_metric╣!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}, "shared_object_id": 14}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
N0
O1"
trackable_list_wrapper
-
P	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
R0
S1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
:╚ (2true_positives
:╚ (2true_negatives
 :╚ (2false_positives
 :╚ (2false_negatives
<
W0
X1
Y2
Z3"
trackable_list_wrapper
-
[	variables"
_generic_user_object
$:"	м2Adam/dense/kernel/m
:2Adam/dense/bias/m
+:)	2Д2Adam/gru/gru_cell/kernel/m
6:4
мД2$Adam/gru/gru_cell/recurrent_kernel/m
):'	Д2Adam/gru/gru_cell/bias/m
$:"	м2Adam/dense/kernel/v
:2Adam/dense/bias/v
+:)	2Д2Adam/gru/gru_cell/kernel/v
6:4
мД2$Adam/gru/gru_cell/recurrent_kernel/v
):'	Д2Adam/gru/gru_cell/bias/v
·2ў
+__inference_sequential_layer_call_fn_636069
+__inference_sequential_layer_call_fn_636674
+__inference_sequential_layer_call_fn_636691
+__inference_sequential_layer_call_fn_636592└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ц2у
F__inference_sequential_layer_call_and_return_conditional_losses_637085
F__inference_sequential_layer_call_and_return_conditional_losses_637496
F__inference_sequential_layer_call_and_return_conditional_losses_636612
F__inference_sequential_layer_call_and_return_conditional_losses_636632└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ш2х
!__inference__wrapped_model_633980┐
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк */в,
*К'
embedding_input         °

╘2╤
*__inference_embedding_layer_call_fn_637503в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_embedding_layer_call_and_return_conditional_losses_637513в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
К2З
2__inference_spatial_dropout1d_layer_call_fn_637518
2__inference_spatial_dropout1d_layer_call_fn_637523
2__inference_spatial_dropout1d_layer_call_fn_637528
2__inference_spatial_dropout1d_layer_call_fn_637533┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ў2є
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637538
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637560
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637565
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637587┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
$__inference_gru_layer_call_fn_637598
$__inference_gru_layer_call_fn_637609
$__inference_gru_layer_call_fn_637620
$__inference_gru_layer_call_fn_637631╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▀2▄
?__inference_gru_layer_call_and_return_conditional_losses_638011
?__inference_gru_layer_call_and_return_conditional_losses_638391
?__inference_gru_layer_call_and_return_conditional_losses_638771
?__inference_gru_layer_call_and_return_conditional_losses_639151╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╨2═
&__inference_dense_layer_call_fn_639160в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_639171в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙B╨
$__inference_signature_wrapper_636657embedding_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 Ч
!__inference__wrapped_model_633980r%&'9в6
/в,
*К'
embedding_input         °

к "-к*
(
denseК
dense         в
A__inference_dense_layer_call_and_return_conditional_losses_639171]0в-
&в#
!К
inputs         м
к "%в"
К
0         
Ъ z
&__inference_dense_layer_call_fn_639160P0в-
&в#
!К
inputs         м
к "К         к
E__inference_embedding_layer_call_and_return_conditional_losses_637513a0в-
&в#
!К
inputs         °

к "*в'
 К
0         °
2
Ъ В
*__inference_embedding_layer_call_fn_637503T0в-
&в#
!К
inputs         °

к "К         °
2┴
?__inference_gru_layer_call_and_return_conditional_losses_638011~%&'OвL
EвB
4Ъ1
/К,
inputs/0                  2

 
p 

 
к "&в#
К
0         м
Ъ ┴
?__inference_gru_layer_call_and_return_conditional_losses_638391~%&'OвL
EвB
4Ъ1
/К,
inputs/0                  2

 
p

 
к "&в#
К
0         м
Ъ ▓
?__inference_gru_layer_call_and_return_conditional_losses_638771o%&'@в=
6в3
%К"
inputs         °
2

 
p 

 
к "&в#
К
0         м
Ъ ▓
?__inference_gru_layer_call_and_return_conditional_losses_639151o%&'@в=
6в3
%К"
inputs         °
2

 
p

 
к "&в#
К
0         м
Ъ Щ
$__inference_gru_layer_call_fn_637598q%&'OвL
EвB
4Ъ1
/К,
inputs/0                  2

 
p 

 
к "К         мЩ
$__inference_gru_layer_call_fn_637609q%&'OвL
EвB
4Ъ1
/К,
inputs/0                  2

 
p

 
к "К         мК
$__inference_gru_layer_call_fn_637620b%&'@в=
6в3
%К"
inputs         °
2

 
p 

 
к "К         мК
$__inference_gru_layer_call_fn_637631b%&'@в=
6в3
%К"
inputs         °
2

 
p

 
к "К         м╝
F__inference_sequential_layer_call_and_return_conditional_losses_636612r%&'Aв>
7в4
*К'
embedding_input         °

p 

 
к "%в"
К
0         
Ъ ╝
F__inference_sequential_layer_call_and_return_conditional_losses_636632r%&'Aв>
7в4
*К'
embedding_input         °

p

 
к "%в"
К
0         
Ъ │
F__inference_sequential_layer_call_and_return_conditional_losses_637085i%&'8в5
.в+
!К
inputs         °

p 

 
к "%в"
К
0         
Ъ │
F__inference_sequential_layer_call_and_return_conditional_losses_637496i%&'8в5
.в+
!К
inputs         °

p

 
к "%в"
К
0         
Ъ Ф
+__inference_sequential_layer_call_fn_636069e%&'Aв>
7в4
*К'
embedding_input         °

p 

 
к "К         Ф
+__inference_sequential_layer_call_fn_636592e%&'Aв>
7в4
*К'
embedding_input         °

p

 
к "К         Л
+__inference_sequential_layer_call_fn_636674\%&'8в5
.в+
!К
inputs         °

p 

 
к "К         Л
+__inference_sequential_layer_call_fn_636691\%&'8в5
.в+
!К
inputs         °

p

 
к "К         о
$__inference_signature_wrapper_636657Е%&'LвI
в 
Bк?
=
embedding_input*К'
embedding_input         °
"-к*
(
denseК
dense         ┌
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637538ИIвF
?в<
6К3
inputs'                           
p 
к ";в8
1К.
0'                           
Ъ ┌
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637560ИIвF
?в<
6К3
inputs'                           
p
к ";в8
1К.
0'                           
Ъ ╖
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637565f8в5
.в+
%К"
inputs         °
2
p 
к "*в'
 К
0         °
2
Ъ ╖
M__inference_spatial_dropout1d_layer_call_and_return_conditional_losses_637587f8в5
.в+
%К"
inputs         °
2
p
к "*в'
 К
0         °
2
Ъ ▒
2__inference_spatial_dropout1d_layer_call_fn_637518{IвF
?в<
6К3
inputs'                           
p 
к ".К+'                           ▒
2__inference_spatial_dropout1d_layer_call_fn_637523{IвF
?в<
6К3
inputs'                           
p
к ".К+'                           П
2__inference_spatial_dropout1d_layer_call_fn_637528Y8в5
.в+
%К"
inputs         °
2
p 
к "К         °
2П
2__inference_spatial_dropout1d_layer_call_fn_637533Y8в5
.в+
%К"
inputs         °
2
p
к "К         °
2