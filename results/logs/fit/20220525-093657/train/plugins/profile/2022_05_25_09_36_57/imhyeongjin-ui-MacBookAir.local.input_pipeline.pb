	L7?A`? @L7?A`? @!L7?A`? @	l??&?l@l??&?l@!l??&?l@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$L7?A`? @???Mb??A???x?& @YZd;?O???*	      7@2U
Iterator::Model::ParallelMapV2{?G?z??!????7?E@){?G?z??1????7?E@:Preprocessing2F
Iterator::Model???Q???!??7??MP@){?G?zt?1????7?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{?G?zt?!????7?5@){?G?zt?1????7?5@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?h?!!Y?B*@)?~j?t?h?1!Y?B*@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9l??&?l@Im??&?tX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Mb?????Mb??!???Mb??      ?!       "      ?!       *      ?!       2	???x?& @???x?& @!???x?& @:      ?!       B      ?!       J	Zd;?O???Zd;?O???!Zd;?O???R      ?!       Z	Zd;?O???Zd;?O???!Zd;?O???b      ?!       JCPU_ONLYYl??&?l@b qm??&?tX@