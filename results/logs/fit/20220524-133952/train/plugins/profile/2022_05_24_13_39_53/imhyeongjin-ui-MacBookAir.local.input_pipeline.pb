	sh??|???sh??|???!sh??|???	9?_-?q@9?_-?q@!9?_-?q@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$sh??|????? ?rh??A?K7?A`??YJ+???*	     ?@@2F
Iterator::Model9??v????!??&?l?C@)9??v????1??&?l?C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!??????@@)?I+???1??????@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;?O??n??!E]t?E;@)?~j?t?x?1/?袋.2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?~j?t?h?!/?袋."@)?~j?t?h?1/?袋."@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no99?_-?q@I??FrTX@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? ?rh???? ?rh??!?? ?rh??      ?!       "      ?!       *      ?!       2	?K7?A`???K7?A`??!?K7?A`??:      ?!       B      ?!       J	J+???J+???!J+???R      ?!       Z	J+???J+???!J+???b      ?!       JCPU_ONLYY9?_-?q@b q??FrTX@