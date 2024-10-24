# NLP

## Datasets

**LexMTurk**    It is composed of 500 instances for English. Each instance contains one sentence from Wikipedia, a target complex word and 50 substitutions
annotated by 50 Amazon Mechanical "turkers". Since each complex word of each instance was annotated by 50 turkers, this dataset owns a pretty good coverage of gold simpliﬁcations.

**BenchLS** It is composed of 929 instances for English, which is from LexMTurk and LSeval (De Belder and Moens 2010). The LSeval contains 429 instances, in which each complex word was annotated by 46 turkers and 9 Ph.D. students. Because BenchLS contains two datasets, it provides the largest range of distinct
complex words amongst the LS dataset of English.

**NNSeval**  It is composed of 239 instances for English, which is a ﬁltered version of BenchLS. These instances of BenchLS are dropped:

* the target complex word in the instance was not regarded as the complex word by a non-native English speaker, and
* any candidate in the instance was regarded as a complex word by a non-native speaker.

Some other files in ./datasets are the methods for using them -- maybe can be utilized someday :)
