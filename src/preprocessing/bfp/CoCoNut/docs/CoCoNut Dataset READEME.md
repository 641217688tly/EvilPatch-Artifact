### Training Data Release

This release contains the data used to train the different CoCoNuT models.
The datasets have been compressed using lrztar. To extract the data, use
`lrzuntar`.

There is one archive per language (Python, C, Java, and JavaScript). 
There is one training set per language, except for Java where we have two different training 
dataset with two different max dates.

These datasets contains raw data extracted from GitHub, GitLab, and Bitbucket and have 
not been shuffled nore tokenized.

Each archive is ordered as:
`language/cutting_year/{add,rem,context,meta}.txt`

The cutting year is the year of the newest commit in the dataset. 
It was decided based on the date of the oldest bug in the test benchmark
(Defects4J, QuixBugs, BugAid, CodeFlaws, and ManyBugs).
We did this to ensure future data is not used in the training (see paper for more information).

Each dataset consists of 4 files:
`add.txt`, `rem.txt`, `context.txt`, and `meta.txt`

There is a line-to-line mapping between these 4 files for each training set.

For example:

5 first lines of rem.txt (i.e., the buggy line/hunk):
```
1 public synchronized StringBuffer append(char ch)
2 ensureCapacity_unsynchronized(count + 1); value[count++] = ch; return this;
3 public String substring(int beginIndex, int endIndex)
4 if (beginIndex < 0 || endIndex > count || beginIndex > endIndex) throw new StringIndexOutOfBoundsException(); if (beginIndex == 0 && endIndex == count) return this; int len = endIndex - beginIndex;  return new String(value, beginIndex + offset, len, (len << 2) >= value.length);
5 public Object next() {
```
5 first lines of add.txt (i.e., the fixed line/hunk):
```
1 public StringBuffer append(Object obj)
2 return append(obj == null ? "null" : obj.toString());
3 public String substring(int begin)
4 return substring(begin, count);
5 public FSEntry next() {
```

This maps to the 5 instances:
```
- public synchronized StringBuffer append(char ch)
+ public StringBuffer append(Object obj)
```

```
- ensureCapacity_unsynchronized(count + 1); value[count++] = ch; return this;
+ return append(obj == null ? "null" : obj.toString());
```

```
- public String substring(int beginIndex, int endIndex)
+ public String substring(int begin)
```

```
- if (beginIndex < 0 || endIndex > count || beginIndex > endIndex) throw new StringIndexOutOfBoundsException(); if (beginIndex == 0 && endIndex == count) return this; int len = endIndex - beginIndex;  return new String(value, beginIndex + offset, len, (len << 2) >= value.length);
+ return substring(begin, count);
```

```
-public Object next() {
+public FSEntry next() { 
```

`context.txt` contains the associated "context". We call context the (in-lined) buggy function (including the buggy lines and comments).
For example, the context of 
```
public synchronized StringBuffer append(char ch)
```
is its associated function:
```
public synchronized StringBuffer append(char ch)  {    ensureCapacity_unsynchronized(count + 1);    value[count++] = ch;    return this;  }
```

This context is used as a secondary input to the context-aware network. The FConv model does not use this context.

`meta.txt` contains some metadata about the project:
```
1056	/local/tlutelli/issta_data/temp/all_java0context/java/2006_temp/2006/1056/68a6301301378680519f2b146daec37812a1bc22/StringBuffer.java/buggy/core/src/classpath/java/java/lang/StringBuffer.java
```
`1056` is the project id. `/local/...` is the absolute path to the buggy file. This can be parsed to extract the commit id: `68a6301301378680519f2b146daec37812a1bc22`, the file name: `StringBuffer.java` and the original path within the project 
`core/src/classpath/java/java/lang/StringBuffer.java`

Unfortunately, we do not kept the mapping between project id and GitHub repository.


### Note about Java 2010 and ensemble:
For the 2010 Java dataset, there are more than 14M instances. Therefore, when building an Ensemble 
approach for this training set, we splitted it into 10 random sub dataset and trained 10 models each with different training inputs.
For other training set. The entire training set was used during training of all ensemble models.



