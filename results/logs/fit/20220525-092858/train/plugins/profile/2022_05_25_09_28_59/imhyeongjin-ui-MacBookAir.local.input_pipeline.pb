	?p=
ף@?p=
ף@!?p=
ף@	q?{????q?{????!q?{????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?p=
ף@?&1???AˡE??? @Y???Q???*	     @P@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Mb??!;?;?R@)???Mb??1;?;?R@:Preprocessing2F
Iterator::Model?~j?t???!c'vb'v2@)?~j?t???1c'vb'v2@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!ى?؉?@)????Mbp?1ى?؉?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!ى?؉?@)????Mb`?1ى?؉?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9q?{????Ij????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?&1????&1???!?&1???      ?!       "      ?!       *      ?!       2	ˡE??? @ˡE??? @!ˡE??? @:      ?!       B      ?!       J	???Q??????Q???!???Q???R      ?!       Z	???Q??????Q???!???Q???b      ?!       JCPU_ONLYYq?{????b qj????X@