	??n??@??n??@!??n??@	d?KJt0??d?KJt0??!d?KJt0??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??n??@???S㥻?AD?l??? @Yy?&1???*	      B@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!?8??8?>@)?I+???1?8??8?>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;?O??n??!      9@);?O??n??1      9@:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1?|?!?q?q3@)y?&1?|?1?q?q3@:Preprocessing2F
Iterator::Model?~j?t???!??????@@){?G?zt?1r?q?+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!??8??8&@)????Mbp?1??8??8&@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9d?KJt0??IZ??.>?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???S㥻????S㥻?!???S㥻?      ?!       "      ?!       *      ?!       2	D?l??? @D?l??? @!D?l??? @:      ?!       B      ?!       J	y?&1???y?&1???!y?&1???R      ?!       Z	y?&1???y?&1???!y?&1???b      ?!       JCPU_ONLYYd?KJt0??b qZ??.>?X@